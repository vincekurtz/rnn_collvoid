#!/usr/bin/env python

##
#
# Create a plot of predicted and actual positions by listening to
# ROS topics that contain pose and covariance information. 
#
##

import rospy
import numpy as np
import GPy
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

# Initialize a rospy node
rospy.init_node("prediction_plotter")

# Number of steps in the future to record predictions for
num_steps = 10

# Global variables store realtime infomation
actual_positions_x = []
actual_positions_y = []
actual_position_time = []

projected_positions = {}
projected_position_time = []

# Wait to get a prediction before starting to record actual pose
got_projected_position = False

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
    global actual_position_time

    
    if (rospy.get_time() > last_time + interval) and got_projected_position:
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        t = data.header.stamp.to_sec()

        actual_positions_x.append(x)
        actual_positions_y.append(y)
        actual_position_time.append(t)

        last_time = rospy.get_time()

def projected_pose_cb(data, step_num):
    """
    Put predicted pose data in a dictionary, index by the time at 
    which the prediction was made
    """
    global projected_positions
    global got_projected_position
    global projected_position_time
   
    # Indicated that we can start collecting real poses
    if not got_projected_position:
        got_projected_position = True

    x = data.pose.pose.position.x
    y = data.pose.pose.position.y
    t = data.header.stamp.to_sec()

    cov = data.pose.covariance

    sigma = np.zeros((2,2))
    sigma[0,0] = cov[0]
    sigma[0,1] = cov[1]
    sigma[1,0] = cov[6]
    sigma[1,1] = cov[7]

    mu = np.array([x,y])

    projected_positions[t] = (mu, sigma)
    
def plot_recorded_data():
    """
    Plot actual and projected positions in position (x,y) space
    """
    # clear the plot
    plt.clf()

    # actual positions
    plt.plot(actual_positions_x, actual_positions_y, 'rx',mew=2)

    # predicted positions
    for t in projected_positions:
        mu, sigma = projected_positions[t]
        x, y = np.random.multivariate_normal(mu, sigma, 100).T

        plt.scatter(x, y, color="blue", alpha=0.01, edgecolors="none")
    
    plt.xlabel("x Position")
    plt.ylabel("y Position")

    plt.savefig('/tmp/xy_predictions.png')
    plt.show()

def plot_recorded_data2(xlim, ylim1, ylim2):
    """
    Plot actual and projected positions as a function of time
    """
    global actual_position_time
    
    plt.clf()  # clear the plot

    # calculate deltas from recorded positions
    delta_ize = lambda lst : np.array([ lst[i+1] - lst[i] for i in range(1, len(lst)-1) ])

    # actual positions
    delta_x = actual_positions_x
    delta_y = actual_positions_y

    # predicted positions
    last_pred_x = None
    last_pred_y = None

    pred_x = []
    pred_y = []
    x_confidence = []
    y_confidence = []

    projected_position_time = sorted(projected_positions.iterkeys())

    print(len(actual_position_time))
    print(len(projected_position_time))

    for t in projected_position_time:
        # traverse the dictionary in order by timestamp
        mu, sigma = projected_positions[t]

        pred_x.append(mu[0])
        pred_y.append(mu[1])
        x_confidence.append(sigma[0][0])
        y_confidence.append(sigma[1][1])

    # calculate predictions
    pred_delta_x = pred_x
    pred_delta_y = pred_y

    # strip first and last elements from times so dimensions match for deltas
    actual_position_time = np.array(actual_position_time)
    projected_position_time = np.array(projected_position_time)

    fig = plt.figure(figsize=(12,8))

    # x on axis 1
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim1)
    ax1.set_ylabel('$\Delta x$')
    ax1.plot(actual_position_time, delta_x, 'kx', label="Data")  # actual data
    ax1.plot(projected_position_time, pred_delta_x, 'b.', label="Mean")
    #ax1.fill_between(projected_position_time, pred_delta_x-x_confidence, pred_delta_x+x_confidence, alpha=0.2, label="Uncertainty Estimate ($100\sigma$)")
    ax1.legend()

    # y on axis 2
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim2)
    ax2.set_ylabel('$\Delta y$')
    ax2.set_xlabel('Time (s)')
    ax2.plot(actual_position_time, delta_y, 'kx', label="Data")  # actual data
    ax2.plot(projected_position_time, pred_delta_y, 'b.', label="Mean")
    #ax2.fill_between(projected_position_time, pred_delta_y-y_confidence, pred_delta_y+y_confidence, alpha=0.2, label="Uncertainty Estimate ($100\sigma$)")
    ax2.legend()

    plt.savefig("/tmp/lstm_pred.png")
    plt.show()

def plot_gp_regression(xlim, ylim1, ylim2):
    """
    Plot actual positions and those predicted with a Gaussian Process model
    """
    global actual_position_time
    
    plt.clf()  # clear the plot

    # calculate deltas from recorded positions
    delta_ize = lambda lst : np.array([ lst[i+1] - lst[i] for i in range(1, len(lst)-1) ])

    # actual positions
    delta_x = delta_ize(actual_positions_x)
    delta_y = delta_ize(actual_positions_y)
    
    X1 = np.array([actual_position_time[1:-1]]).T
    X2 = np.array([actual_position_time[1:-1]]).T
    Y1 = np.array([delta_x]).T
    Y2 = np.array([delta_y]).T


    K = GPy.kern.RBF(input_dim=1)
    icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=2, kernel=K)
    m = GPy.models.GPCoregionalizedRegression([X1,X2],[Y1,Y2], kernel=icm)
    m.optimize()

    # Plot stuff!
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim1)
    ax1.set_ylabel('$\Delta x$')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,0)],which_data_rows=slice(0,100),ax=ax1)
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim2)
    ax2.set_ylabel('$\Delta y$')
    ax2.set_xlabel('time (s)')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,1)],which_data_rows=slice(100,200),ax=ax2)

    plt.savefig("/tmp/gp_pred.png")
    plt.show()
    

# Start subscribers
rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, real_pose_cb)

for i in range(1,num_steps+1):
    topic_name = "/robot_0/predicted_pose/step_%d" % i
    rospy.Subscriber(topic_name, PoseWithCovarianceStamped, projected_pose_cb, callback_args=i)

while not rospy.is_shutdown():
    rospy.spin()

# plot stuff
#plot_gp_regression(xlim=(0,10), ylim1=(-0.8, 0.8), ylim2=(-0.2, 0.2))
#plot_recorded_data2(xlim=(0,10), ylim1=(-0.8,0.8), ylim2=(-0.2,0.2))
plot_recorded_data()
#plot_recorded_data2(xlim=(50,60), ylim1=(-0.8,0.8), ylim2=(-0.2,0.2))
