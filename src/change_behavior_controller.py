#!/usr/bin/env python

##
#
# Simple controller that changes behavior after some time. For testing
# the effectiveness of our approach on obstacles that don't display the
# same type of behavior forever.
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

    # Set parameters of motion
    speed = 1   # How fast to move forward and back
    duration = 1  # How long to move each way (in seconds)


    while not rospy.is_shutdown():
        # First behavior: oscillating in x, drifing in y
        cmd_vel = Twist()
        cmd_vel.linear.y = -0.5
        for j in range(5):
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

        # Second behavior: oscillating in y, drifting in x
        cmd_vel = Twist()
        cmd_vel.linear.x = -0.5
        for j in range(100):
            for i in range(duration*hz / 2):
                # Move forward...
                cmd_vel.linear.y = speed
                controller.publish(cmd_vel)
                rate.sleep()
            for i in range(duration*hz / 2):
                # ... and back
                cmd_vel.linear.y = -speed
                controller.publish(cmd_vel)
                rate.sleep()

except rospy.ROSInterruptException:
    # Quit gracefully with ^C
    pass

