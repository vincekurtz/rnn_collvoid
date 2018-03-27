#!/usr/bin/env python

##
#
# Simple controller to move a robot back and forth
#
##

import rospy
from geometry_msgs.msg import Twist

# The topic we'll use to control the robot
command_topic = '/robot_0/cmd_vel'

try:
    # Initialize node and publisher
    rospy.init_node('oscillating_controller', anonymous=True)
    controller = rospy.Publisher(command_topic, Twist, queue_size=10)
    hz = 10  # refresh rate in Hz
    rate = rospy.Rate(hz)
    cmd_vel = Twist()

    # Set parameters of motion
    speed = 1   # How fast to move forward and back
    duration = 1  # How long to move each way (in seconds)

    # Set a constant y velocity
    cmd_vel.linear.y = 0.1

    while not rospy.is_shutdown():
        for i in range(duration*hz):
            # Move forward...
            cmd_vel.linear.x = speed
            controller.publish(cmd_vel)
            rate.sleep()
        for i in range(duration*hz):
            # ... and back
            cmd_vel.linear.x = -speed
            controller.publish(cmd_vel)
            rate.sleep()

except rospy.ROSInterruptException:
    # Quit gracefully with ^C
    pass

