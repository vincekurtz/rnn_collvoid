#!/usr/bin/env python

##
#
# A simple controller that publishes to ~/cmd_vel_nominal
#
##

import sys
import rospy
import math
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry

if len(sys.argv) < 4:
    print("Usage: python %s [robot_name] [goal_x] [goal_y]" % sys.argv[0])
    sys.exit(1)

# get the robot name and a goal position from the command line
robot_name = sys.argv[1]
goal = Pose()
goal.position.x = float(sys.argv[2])
goal.position.y = float(sys.argv[3])

# global variable to keep track of obstacle pose
pose = Pose()

def odom_callback(data):
    global pose
    pose = data.pose.pose

# initialize the node, publisher, subscriber
rospy.init_node("simple_controller", anonymous=True)
pub = rospy.Publisher(robot_name + '/cmd_vel_nominal', Twist, queue_size=10)
odom = rospy.Subscriber(robot_name + '/base_pose_ground_truth', Odometry, odom_callback)

def proportional_controller(pose, goal):
    """Simple proportional controller, returns a command velocity"""
    vx_max = 0.7
    vy_max = 0.7

    kp = 1

    vx = -kp*(pose.position.x - goal.position.x)
    vy = -kp*(pose.position.y - goal.position.y)

    vx = bound_abs(vx, vx_max)
    vy = bound_abs(vy, vy_max)

    cmd_vel = Twist()
    cmd_vel.linear.x = vx
    cmd_vel.linear.y = vy

    return cmd_vel

def bound_abs(val, maximum):
    return max(-maximum, min(maximum,val))

def dist(pose1, pose2):
    d_squared = (pose1.position.x - pose2.position.x)**2 + (pose1.position.y - pose2.position.y)**2
    return math.sqrt(d_squared)

# go to the given position
while (not rospy.is_shutdown()) and (dist(pose,goal) > 0.01):
    cmd_vel = proportional_controller(pose, goal)
    pub.publish(cmd_vel)

    rospy.sleep(0.1)

