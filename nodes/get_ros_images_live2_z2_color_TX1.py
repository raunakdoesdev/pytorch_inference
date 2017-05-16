#!/usr/bin/env python
"""
reed to run roslaunch first, e.g.,

roslaunch bair_car bair_car.launch use_zed:=true record:=false
"""

weight_file_path = '/home/nvidia/catkin_ws/src/bair_car/nodes/weights'

# Labels
Direct = 1.
Follow = 0.
Play = 0.
Furtive = 0.
Caf = 0.0
Racing = 0.0

motor_gain = 1.0
steer_gain = 1.0

last_time = 0

verbose = False

nframes = 2 # default superseded by net

# try:
from kzpy3.utils import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from nets.z2_color_batchnorm import Z2ColorBatchNorm
from nets.z2_color_batchnorm import *

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def init_model():
    global solver, scale, nframes
    # Load PyTorch model
    save_data = torch.load(weight_file_path)
    # Initializes Solver
    solver = Z2ColorBatchNorm().cuda()
    solver.load_state_dict(save_data['net'])
    solver.eval()
    nframes = solver.N_FRAMES

    # Create scaling layer
    scale = nn.AvgPool2d(kernel_size=3, stride=2, padding=1).cuda()

init_model()

@static_vars(torch_motor_previous=49, torch_steer_previous=49)
def run_model(input, metadata):
    """
    Runs neural net to get motor and steer data. Scales output to 0 to 100 and applies an IIR filter to smooth the
    performance.

    :param input: Formatted input data from ZED depth camera
    :param metadata: Formatted metadata from user input
    :return: Motor and Steering values
    """
    
    output = solver(input, Variable(metadata))  # Run the neural net

    # Get latest prediction
    torch_motor = 100 * output[0][19].data[0]
    torch_steer = 100 * output[0][9].data[0]

    if verbose:
        print('Torch Prescale Motor: ' + str(torch_motor))
        print('Torch Prescale Steer: ' + str(torch_steer))
    
    # Scale Output
    torch_motor = int((torch_motor - 49.) * motor_gain + 49.)
    torch_steer = int((torch_steer - 49.) * steer_gain + 49.)

    # Bound the output
    torch_motor = max(0, torch_motor)
    torch_steer = max(0, torch_steer)
    torch_motor = min(99, torch_motor)
    torch_steer = min(99, torch_steer)

    # Apply an IIR Filter
    torch_motor = int((torch_motor + run_model.torch_motor_previous) / 2.0)
    run_model.torch_motor_previous = torch_motor
    torch_steer = int((torch_steer + run_model.torch_steer_previous) / 2.0)
    run_model.torch_steer_previous = torch_steer

    return torch_motor, torch_steer

def format_camera_data(left_list, right_list):
    """
    Formats camera data from raw inputs from camera.

    :param l0: left camera data from time step 0
    :param l1: right camera data from time step 1
    :param r0: right camera dataa from time step 0
    :param r1: right camera data from time step 0
    :return: formatted camera data ready for input into pytorch z2color
    """

    camera_start = time.clock()
    
    listoftensors = []
    for side in (left_list, right_list):
        for i in range(nframes):
            listoftensors.append(torch.from_numpy(side[-i - 1]))
            
    camera_data = torch.cat(listoftensors, 2)

    camera_data = camera_data.cuda().float()/255. - 0.5
    camera_data = torch.transpose(camera_data, 0, 2)
    camera_data = torch.transpose(camera_data, 1, 2)
    camera_data = camera_data.unsqueeze(0)
    camera_data = scale(Variable(camera_data))
    camera_data = scale(camera_data)

    return camera_data


def format_metadata(raw_metadata):
    """
    Formats meta data from raw inputs from camera.
    :return:
    """
    metadata = torch.FloatTensor()
    for mode in raw_metadata:
        metadata = torch.cat((torch.FloatTensor(1, 13, 26).fill_(mode), metadata), 0)
    return metadata.cuda().unsqueeze(0)

#
########################################################


########################################################
#          ROSPY SETUP SECTION
import roslib
import std_msgs.msg
import geometry_msgs.msg
import cv2
from cv_bridge import CvBridge,CvBridgeError
import rospy
from sensor_msgs.msg import Image
bridge = CvBridge()
rospy.init_node('listener',anonymous=True)

left_list = []
right_list = []
A = 0
B = 0
state = 0
previous_state = 0
state_transition_time_s = 0

def state_callback(data):
    global state, previous_state
    data.data = 6
    if state != data.data:
        if state in [3,5,6,7] and previous_state in [3,5,6,7]:
            pass
        else:
            previous_state = state
    state = data.data
def right_callback(data):
    global A,B, left_list, right_list, solver
    A += 1
    cimg = bridge.imgmsg_to_cv2(data,"bgr8")
    if len(right_list) > nframes + 3:
        right_list = right_list[-(nframes + 3):]
    right_list.append(cimg)
def left_callback(data):
    global A,B, left_list, right_list
    B += 1
    cimg = bridge.imgmsg_to_cv2(data,"bgr8")
    if len(left_list) > nframes + 3:
        left_list = left_list[-(nframes + 3):]
    left_list.append(cimg)
def state_transition_time_s_callback(data):
    global state_transition_time_s
    state_transition_time_s = data.data



freeze = False

##
########################################################

import thread
import time


rospy.Subscriber("/bair_car/zed/right/image_rect_color",Image,right_callback,queue_size = 1)
rospy.Subscriber("/bair_car/zed/left/image_rect_color",Image,left_callback,queue_size = 1)
rospy.Subscriber('/bair_car/state', std_msgs.msg.Int32,state_callback)
rospy.Subscriber('/bair_car/state_transition_time_s', std_msgs.msg.Int32, state_transition_time_s_callback)
steer_cmd_pub = rospy.Publisher('cmd/steer', std_msgs.msg.Int32, queue_size=100)
motor_cmd_pub = rospy.Publisher('cmd/motor', std_msgs.msg.Int32, queue_size=100)
freeze_cmd_pub = rospy.Publisher('cmd/freeze', std_msgs.msg.Int32, queue_size=100)
model_name_pub = rospy.Publisher('/bair_car/model_name', std_msgs.msg.String, queue_size=10)

ctr = 0

time_step = Timer(1)
caffe_enter_timer = Timer(1)
folder_display_timer = Timer(30)
git_pull_timer = Timer(60)
reload_timer = Timer(10)
torch_steer_previous = 49
torch_motor_previous = 49

while not rospy.is_shutdown():
    if state in [3,5,6,7]:
        if (previous_state not in [3,5,6,7]):
            previous_state = state
            caffe_enter_timer.reset()
        if not caffe_enter_timer.check():
            print "waiting before entering caffe mode..."
            steer_cmd_pub.publish(std_msgs.msg.Int32(49))
            motor_cmd_pub.publish(std_msgs.msg.Int32(49))
            time.sleep(0.1)
            continue
        else:
            if len(left_list) > nframes + 2:
                camera_data = format_camera_data(left_list, right_list)
                metadata = format_metadata((Racing, 0, Follow, Direct, Play, Furtive))

                torch_motor, torch_steer = run_model(camera_data, metadata)

                freeze_cmd_pub.publish(std_msgs.msg.Int32(freeze))
                
                cur_time = time.time()
                print(cur_time - last_time)
                print(str(1./(cur_time - last_time)) + ' Hz')
                last_time = cur_time

                if state in [3,6]:          
                    steer_cmd_pub.publish(std_msgs.msg.Int32(torch_steer))
                if state in [6,7]:
                    motor_cmd_pub.publish(std_msgs.msg.Int32(torch_motor))
                if verbose:
                    print torch_motor,torch_steer,motor_gain,steer_gain,state

    else:
        caffe_enter_timer.reset()
    if time_step.check():
        print(d2s("In state",state,"for",state_transition_time_s,"seconds, previous_state =",previous_state))
        time_step.reset()
