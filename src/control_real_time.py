#!/usr/bin/env python

##
# 
#  Load a trained model (generated by train_lstm.py) and
#  use it to control a different agent #
##

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from data_container import DataContainer
from network_variables import *
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

base_dir = "/home/vjkurtz/catkin_ws/src/rnn_collvoid" 

# Store real-time data to access asynchronously
velocity_data = Odometry().twist.twist.linear  # linear x,y,z velocities
position_data = Odometry().pose.pose.position  # x,y,z position

my_position_data = Odometry().pose.pose.position  # x,y,z position

def predict_distribution(observations, num_samples, sess):
    """
    Given a sequence of observations, use dropout to generate a gaussian
    distribution over the predictions. 

    Returns a mean vector and covariance matrix based on num_samples
    different possible outputs (obtained with different dropout masks)

    Dropout probabilities are defined in network_variables.py
    """
    
    predictions = []

    for i in range(num_samples):
        # This gives a (num_steps x batch_size x output_size), ie (100 x 1 x 4), numpy array. 
        all_pred = sess.run(predicted_outputs, { inputs: observations })
        # We're really only interested in the last prediction: the one for the next step
        next_pred = all_pred[-1][0]   # [deltax, deltay, xdot1, ydot1] 
        
        predictions.append(next_pred)

    predictions = np.asarray(predictions)  # convert to np array

    # use sample mean and covariance, which are MLE assuming i.i.d. samples
    # from a Gaussian distribution
    mu = np.mean(predictions, axis=0)
    sigma = np.cov(predictions.T)

    return (mu, sigma)

def odom_callback(data):
    """
    Odometry message callback for the obstacle
    """
    global velocity_data
    global position_data
    velocity_data = data.twist.twist.linear
    position_data = data.pose.pose.position

def my_odom_callback(data):
    """
    Odometry message callback for the vehicle we control
    """
    global my_position_data
    my_position_data = data.pose.pose.position

def get_first_obs():
    """
    Get the initial set of observations. 
    return this initial set of observations, and the initial position
    """
    x = position_data.x
    y = position_data.y

    meas = get_latest_meas(x,y)

    # observations must be a (N, 1, 4) np array
    observations = np.array([[ meas ]])

    return(x,y,observations)

def get_latest_meas(last_x, last_y):
    """
    Return the latest velocity/position measurements: [deltax, deltay, xdot, ydo]
    """
    deltax = position_data.x - last_x
    deltay = position_data.y - last_y
    xdot = velocity_data.x
    ydot = velocity_data.y

    return np.array([deltax, deltay, xdot, ydot])

def control_robot_2(mu, sigma, v_desired, publisher):
    """
    Given a distribution of likely obstacle positions and a desired velocity,
    move the robot in a way that minimizes deviation from the desired velocity
    subject to a constraint on the probability of collision. 
    """
    
    dt = 0.1  # TODO cleanup
    obs_radius = 0.25
    my_radius = 0.25

    # Calculate desired position TODO: take orientation into account
    xdes = [my_position_data.x + (v_desired.linear.x * dt), my_position_data.y + (v_desired.linear.y * dt)]

    # Construct a random variable representing probable obstacle positions
    xobs = [position_data.x, position_data.y]
    rv = multivariate_normal([xobs[0] + mu[0], xobs[1] + mu[1]], sigma[0:2, 0:2])

    # Calculate (approximate) probibility of collision
    rad = obs_radius + 2*my_radius
    top_right = lambda x : [x[0] + rad, xdes[1] + rad]
    bottom_right = lambda x : [x[0] + rad, xdes[1] - rad]
    top_left = lambda x : [x[0] - rad, xdes[1] + rad]
    bottom_left = lambda x :[x[0] - rad, xdes[1] - rad]

    coll_prob = lambda x : rv.cdf(top_right(x)) - rv.cdf(bottom_right(x)) - rv.cdf(top_left(x)) + rv.cdf(bottom_left(x))
    performance_cost = lambda x : (x[0] - xdes[0])**2 + (x[1]-xdes[1])**2

    # Calculate an optimal next position
    theta = 0.1   # maximum allowable probibility of collision
    cons = ({'type': 'ineq', 'fun': lambda x : theta - coll_prob(x)})
    res = minimize(performance_cost, xdes, constraints=cons, options={"maxiter": 900}, method="COBYLA")
    best_x = res.x

    if not res.success:
        print("Unable to find optimal solution!")
        print(res)
    print(best_x, performance_cost(best_x), coll_prob(best_x))

    # translate position into a command velocity
    cmd_vel = Twist()
    best_x_vel = (best_x[0] - my_position_data.x)/dt
    best_y_vel = (best_x[1] - my_position_data.y)/dt
    cmd_vel.linear.x = best_x_vel   # weird conversion between reference frames again TODO cleanup
    cmd_vel.linear.y = best_y_vel

    publisher.publish(cmd_vel)


def control_robot(mu, sigma, v_desired, publisher):
    """
    Given a distribution of likely obstacle positions at the next step, move the robot in this direction

    TODO: 
        - clean up this function!
        - use a more systematic velocity-obstacle-derived approach
        - handle arbitrary robot orientations
        - Use consistent representations of velocities and positions
    """
    dt = 0.1   # TODO cleanup
    # TODO add a maximum velocity constraint (?)

    # Desired position from the desired velocity
    # TODO positions and velocities aren't in the same reference frame: will need to clean this up
    xdes = [my_position_data.x + (v_desired.linear.x * dt), my_position_data.y + (v_desired.linear.y * dt)]

    obs_projected_position_x = position_data.x + mu[0]
    obs_projected_position_y = position_data.y + mu[1]

    obs_projected_position = [obs_projected_position_x, obs_projected_position_y]

    alpha = 1
    beta = 1

    # TODO: update collision cost to be a proper probability of collision
    corrected_covariance = sigma[0:2,0:2] * 1000   # increase variance for robustness
    collision_cost = lambda x : alpha * multivariate_normal.pdf(x, mean=obs_projected_position, cov=corrected_covariance)
    performance_cost = lambda x : beta * ( (x[0] - xdes[0])**2 + (x[1] - xdes[1])**2 )   # distance from the desired position
    total_cost = lambda x : collision_cost(x) + performance_cost(x)

    best_x = minimize(total_cost, xdes).x  # initial guess of best next position is the desired position

    # translate position into a command velocity
    best_x_vel = (best_x[0] - my_position_data.x)/dt
    best_y_vel = (best_x[1] - my_position_data.y)/dt

    cmd_vel = Twist()
    cmd_vel.linear.x = best_x_vel   # weird conversion between reference frames again TODO cleanup
    cmd_vel.linear.y = best_y_vel

    publisher.publish(cmd_vel)

def position_to_velocity(position, dt):
    pass

def velocity_to_position(velocity, dt):
    pass

def main():

    tao = 0.1  # time between samples, in seconds
    buffer_length = 20   # number of observations to predict based on
    num_samples = 10     # number of passes used to approximate the distribution of predicted next position
        
    rospy.init_node('rnn_observer_controller')
    odometer = rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, odom_callback)
    my_odometer = rospy.Subscriber("/robot_1/base_pose_ground_truth", Odometry, my_odom_callback)
    controller = rospy.Publisher("/robot_1/cmd_vel", Twist, queue_size=10)

    # get initial positions and observations
    x, y, observations = get_first_obs()

    # set desired velocity
    des_vel = Twist()

    rospy.sleep(tao)

    with tf.Session() as sess:  # start up the tensorflow session
        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        # Load saved session
        saver = tf.train.Saver()
        saver.restore(sess, "%s/tmp/LSTM_saved_model" % base_dir)


        while not rospy.is_shutdown():
            # Get latest measurments
            meas = get_latest_meas(x,y)
            x += meas[0]    # update the latest positions
            y += meas[1]

            # Update the observation buffer
            if len(observations) < buffer_length:
                # simply add to the buffer
                observations = np.append(observations, [[ meas ]], axis=0)
            else:
                # update with replacement
                observations = np.append(observations[1:], [[ meas ]], axis=0)

            # Generate a prediction
            mu, sigma = predict_distribution(observations, num_samples, sess)

            # Control the robot
            des_vel.linear.y = -1
            control_robot_2(mu, sigma, des_vel, controller)

            rospy.sleep(tao)


if __name__=="__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
