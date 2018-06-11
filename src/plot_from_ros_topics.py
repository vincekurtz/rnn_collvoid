#!/usr/bin/env python

##
#
# Create a plot of predicted and actual positions by listening to
# ROS topics that contain pose and covariance information. 
#
##

import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

# Initialize a rospy node
rospy.init_node("prediction_plotter")

# Global variables store realtime infomation
actual_positions_x = []
actual_positions_y = []
projected_positions = { 1:[], 2:[], 3:[], 4:[] }

# Parameters for timed callbacks
r = 10  # recording rate of messages, in Hz
rate = rospy.Rate(r)
interval = 1. / r
last_time = rospy.get_time()

# Callback functions
def real_pose_cb(data):
    """
    Handle the actual position information from Stage
    """
    global last_time
    global actual_positions_x
    global actual_positions_y

    if (rospy.get_time() > last_time + interval):
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y

        actual_positions_x.append(x)
        actual_positions_y.append(y)

        last_time = rospy.get_time()

def projected_pose_cb(data, step_num):
    global last_time
    global projected_positions

    x = data.pose.pose.position.x
    y = data.pose.pose.position.y

    cov = data.pose.covariance

    sigma = np.zeros((2,2))
    sigma[0,0] = cov[0]
    sigma[0,1] = cov[1]
    sigma[1,0] = cov[6]
    sigma[1,1] = cov[7]

    mu = np.array([x,y])

    projected_positions[step_num].append((mu, sigma))
    
def plot_recorded_data():
    """
    Plot actual and projected positions that we've recorded
    """
    # actual positions
    plt.plot(actual_positions_x, actual_positions_y, 'rx')

    # predicted positions
    for i in projected_positions:
        for (mu, sigma) in projected_positions[i]:
            print(mu)
            print(sigma)
            x, y = np.random.multivariate_normal(mu, sigma, 50).T

            plt.scatter(x, y, color="blue", alpha=0.1, edgecolors="none")

    plt.show()

# Start subscribers
rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, real_pose_cb)
rospy.Subscriber("/robot_0/predicted_pose/step_1", PoseWithCovarianceStamped, projected_pose_cb, callback_args=1)
rospy.Subscriber("/robot_0/predicted_pose/step_2", PoseWithCovarianceStamped, projected_pose_cb, callback_args=2)
rospy.Subscriber("/robot_0/predicted_pose/step_3", PoseWithCovarianceStamped, projected_pose_cb, callback_args=3)
rospy.Subscriber("/robot_0/predicted_pose/step_4", PoseWithCovarianceStamped, projected_pose_cb, callback_args=4)

while not rospy.is_shutdown():
    rospy.spin()

# plot stuff
plot_recorded_data()
