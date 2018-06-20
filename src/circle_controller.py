#!/usr/bin/env python

##
#
# Simple controller to move a robot in a regular circle
#
##

import rospy
from geometry_msgs.msg import Twist
import numpy as np

# The topic we'll use to control the robot
command_topic = '/robot_0/cmd_vel'

try:
    # Initialize node and publisher
    rospy.init_node('oscillating_controller', anonymous=True)
    controller = rospy.Publisher(command_topic, Twist, queue_size=10)
    hz = 100  # refresh rate in Hz
    rate = rospy.Rate(hz)
    cmd_vel = Twist()

    # motion parameters
    A = 1
    w = 1

    while not rospy.is_shutdown():
        t = rospy.get_time()   # time in seconds
        cmd_vel.linear.x = A*np.sin(w*t)
        cmd_vel.linear.y = A*np.cos(w*t)

        controller.publish(cmd_vel)
        rate.sleep()

except rospy.ROSInterruptException:
    # Quit gracefully with ^C
    pass

