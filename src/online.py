#!/usr/bin/env python

##
#
# Train a simple LSTM RNN to predict future
# positions based on current position and velocity. 
#
#
##

import csv
import numpy as np
import tensorflow as tf
import rospy
from nav_msgs.msg import Odometry
import threading


# TRAINING DATA: these variables are updated in real time
last_x = None
last_y = None
position_history = None

last_time = None  # so we only keep track of data so often

def odom_callback(data):
    """
    Updates the globally stored training data
    """
    global position_history
    global last_x  # x position
    global last_y  # y position
    global last_time

    if last_time is None:
        # initialize the time counter: this must happen
        # after the node an everything is started
        last_time = rospy.get_time()

    if (rospy.get_time() > last_time + TIMESTEP):
        curr_x = data.pose.pose.position.x
        curr_y = data.pose.pose.position.y

        if last_x is not None:
            deltax = curr_x - last_x
            deltay = curr_y - last_y

            if position_history is not None:
                position_history = np.vstack((position_history, np.array([[[deltax, deltay]]])))
            else:
                position_history = np.array([[[deltax, deltay]]])

        last_x = curr_x
        last_y = curr_y
        last_time = rospy.get_time()

def get_io_data(pos_hist):
    """
    Given a history of delta_x and delta_y's, return input
    and output data that can be used to train the network.
    """
    N = len(pos_hist)
    ipt = pos_hist[0:N-1]
    opt = pos_hist[1:N]

    return(ipt, opt)

class OnlineLSTMNetwork():
    def __init__(self, scope):
        with tf.variable_scope(scope):
            TIMESTEP = 0.1   # seconds between samples
            INPUT_SIZE = 2   # last changes 2D change in position
            OUTPUT_SIZE = 2   # Next 2D change in position
            RNN_HIDDEN = 100
            LEARNING_RATE = 0.003

            self.inputs = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))
            self.outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))

            # Create a basic LSTM cell, there are other options too
            cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)

            # Add dropout
            self.cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell,
                    input_keep_prob=0.9,
                    output_keep_prob=0.9,
                    state_keep_prob=0.9,
                    variational_recurrent=False,
                    input_size=INPUT_SIZE,
                    dtype=tf.float32,
                    seed=None
            )

            # Create initial state as all zeros
            batch_size = 1  # because online learning
            self.initial_state = self.cell.zero_state(batch_size, tf.float32)

            # Given a set of inputs, return a tuple with rnn outputs and rnn state
            self.rnn_outputs, self.rnn_states = tf.nn.dynamic_rnn(self.cell, self.inputs, initial_state=self.initial_state, time_major=True)

            # Project rnn outputs to our OUTPUT_SIZE
            self.final_projection = lambda x: tf.contrib.layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=None)
            self.predicted_outputs = tf.map_fn(self.final_projection, self.rnn_outputs)

            # Compute the error that we want to minimize
            self.error = tf.losses.huber_loss(self.outputs, self.predicted_outputs)
            #error = tf.losses.absolute_difference(outputs, predicted_outputs)

            # Optimization
            self.train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.error)

            # Accuracy measurment
            accuracy = tf.reduce_mean(tf.abs(outputs - predicted_outputs))

    def predict(session):
        rate = rospy.Rate(1.0/TIMESTEP)
        while not rospy.is_shutdown():
            correct_output = position_history[-1]
            X, Y = get_io_data(position_history)
            pred_output = session.run(predicted_outputs, { inputs: X })[-1]

            xerr = float("{0:.4f}".format(correct_output[0,0]-pred_output[0,0]))
            yerr = float("{0:.4f}".format(correct_output[0,1]-pred_output[0,1]))
            print(xerr, yerr)

            rate.sleep()

    def train(session):
        NUM_EPOCHS = 100

        for _ in range(10):
            # Get input/output data from our position history
            X, Y = get_io_data(position_history)

            for epoch in range(NUM_EPOCHS):
                total_error = 0
                # train_fn triggers backprop
                total_error += session.run([error, train_fn], { inputs: X, outputs: Y})[0]

            total_error /= NUM_EPOCHS

            print("Train error: %.6f" % (total_error))

if __name__=="__main__":
    try:
        rospy.init_node('rnn_data_collector')
        odom = rospy.Subscriber('/robot_0/base_pose_ground_truth', Odometry, odom_callback)
        rospy.sleep(1)  # wait a second for things to initialize

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            train(session)

            #t1 = threading.Thread(target=predict, args=(session,))
            #thread.start_new_thread(test1, (session,))
            #thread.start_new_thread(predict, (session,))

    except rospy.ROSInterruptException:
        pass
