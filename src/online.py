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

import matplotlib.pyplot as plt

TIMESTEP = 0.1   # seconds between samples

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
    def __init__(self, coord):
        # The coordinator will tell separate threads when to stop
        self.coord = coord

        # Set up two tensorflow graphs for training and prediction
        self.train_graph = tf.Graph()
        self.pred_graph = tf.Graph()
        
        self.train_device = '/cpu:0'
        self.pred_device = '/cpu:0'

        # Specify where to save checkpoints
        self.checkpoint_location = "/tmp/graph.checkpoint"

        # Keep track of when a new update to the model is ready
        self.update_ready = False

    def build_net(self, graph, device, test=False):
        """
        Create a tensorflow graph that represents our network. Some parameters
        may change slightly depending whether this is a graph for training or for 
        testing (for example, whether we use dropout for regularization or variational
        inference). 
        """
        with graph.as_default():
            with graph.device(device):
                INPUT_SIZE = 2   # last changes 2D change in position
                OUTPUT_SIZE = 2   # Next 2D change in position
                RNN_HIDDEN = 100
                LEARNING_RATE = 0.003

                inputs = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))
                outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))

                # Create a basic LSTM cell, there are other options too
                cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)

                var_recurr=test  # if it's the test set, use variational inference style dropout.
                                 # Otherwise, use normal dropout for regularization

                # Add dropout
                cell = tf.nn.rnn_cell.DropoutWrapper(
                        cell,
                        input_keep_prob=0.9,
                        output_keep_prob=0.9,
                        state_keep_prob=0.9,
                        variational_recurrent=var_recurr,
                        input_size=INPUT_SIZE,
                        dtype=tf.float32,
                        seed=None
                )

                # Create initial state as all zeros
                batch_size = 1  # because online learning
                initial_state = cell.zero_state(batch_size, tf.float32)

                # Given a set of inputs, return a tuple with rnn outputs and rnn state
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

                # Project rnn outputs to our OUTPUT_SIZE
                final_projection = lambda x: tf.contrib.layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=None)
                predicted_outputs = tf.map_fn(final_projection, rnn_outputs)

                # Compute the error that we want to minimize
                error = tf.losses.huber_loss(outputs, predicted_outputs)
                #error = tf.losses.absolute_difference(outputs, predicted_outputs)

                # Optimization
                train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

                # Accuracy measurment
                accuracy = tf.reduce_mean(tf.abs(outputs - predicted_outputs))

                # Saving function for checkpoint creation
                saver = tf.train.Saver()

                return tf.global_variables_initializer(), inputs, outputs, predicted_outputs, error, train_fn, saver

    def predict(self):
        """
        Predict the next position of an obstacle based on the latest
        observed history.

        Global variable TIMESTEP must be set.
        """
        t = 0  # keep track of time for plotting
        try:
            rate = rospy.Rate(1.0/TIMESTEP)

            # Set up our graph
            init, inputs, outputs, predicted_outputs, error, train_fn, saver = self.build_net(self.pred_graph, self.pred_device)

            with tf.Session(graph=self.pred_graph) as sess:
                saver.restore(sess, self.checkpoint_location)

                while not self.coord.should_stop():
                    correct_output = position_history[-1]
                    X, Y = get_io_data(position_history)
                    pred_output = sess.run(predicted_outputs, { inputs: X })[-1]

                    xerr = float("{0:.4f}".format(abs(correct_output[0,0]-pred_output[0,0])))
                    yerr = float("{0:.4f}".format(abs(correct_output[0,1]-pred_output[0,1])))

                    # Dynamically update the plot
                    plt.scatter(t, yerr, color='red')
                    plt.scatter(t, xerr, color='blue')
                    plt.pause(1e-9)

                    if self.update_ready:
                        # Update the network parameters every so often
                        saver.restore(sess, self.checkpoint_location)
                        self.update_ready = False

                    rate.sleep()

                    t += TIMESTEP  # update the time for plotting

        except rospy.ROSInterruptException:
            self.coord.request_stop()

    def train(self):
        """
        Train the network repeatedly, using the given
        number of epochs.
        """
        try:
            NUM_EPOCHS = 100

            # Set up our graph
            init, inputs, outputs, predicted_outputs, error, train_fn, saver = self.build_net(self.train_graph, self.train_device, test=True)

            with tf.Session(graph=self.train_graph) as sess:
                sess.run(init)
                while not self.coord.should_stop():
                    # Get input/output data from our position history
                    X, Y = get_io_data(position_history)

                    for epoch in range(NUM_EPOCHS):
                        total_error = 0
                        # train_fn triggers backprop
                        total_error += sess.run([error, train_fn], { inputs: X, outputs: Y})[0]

                    total_error /= NUM_EPOCHS

                    # Save a checkpoint
                    saver.save(sess, self.checkpoint_location)
                    print("CP saved. Train error: %.6f" % (total_error))
                    self.update_ready = True   # signal that we're ready to use this new data

        except rospy.ROSInterruptException:
            self.coord.request_stop()

if __name__=="__main__":
    coord = tf.train.Coordinator()
    try:
        rospy.init_node('rnn_data_collector')
        odom = rospy.Subscriber('/robot_0/base_pose_ground_truth', Odometry, odom_callback)
        rospy.sleep(1)  # wait a second for things to initialize

        nn = OnlineLSTMNetwork(coord)
        train = threading.Thread(target=nn.train)
        predict = threading.Thread(target=nn.predict)

        train.start()
        predict.start()

        coord.join([train, predict])

    except rospy.ROSInterruptException:
        pass
