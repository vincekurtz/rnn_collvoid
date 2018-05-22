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
import traceback

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

class LSTMNetwork():
    """
    A representation of an LSTM network, using the 
    specified graph
    """
    def __init__(self, graph, device, variational_recurrent=False):
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

                # Add dropout
                cell = tf.nn.rnn_cell.DropoutWrapper(
                        cell,
                        input_keep_prob=0.9,
                        output_keep_prob=0.9,
                        state_keep_prob=0.9,
                        variational_recurrent=variational_recurrent,
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

                # Set key network attributes
                self.init = tf.global_variables_initializer()
                self.inputs = inputs
                self.outputs = outputs
                self.predicted_outputs = predicted_outputs
                self.error = error
                self.train_fn = train_fn
                self.saver = saver


class OnlineNetworkTrainer():
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
       
        # Update rate
        self.rate = rospy.Rate(1.0/TIMESTEP)

    def test(self):
        """
        Predict the current position of an obstacle based on the latest
        observed history.
        """
        t = 0.0  # keep track of time for plotting

        # plot parameters
        plt.xlabel("time (s)")
        plt.ylabel("Absolute Error (rolling average)")
        xerr_lst = []
        yerr_lst = []

        try:

            # Set up our graph
            nn = LSTMNetwork(self.pred_graph, self.pred_device, variational_recurrent=True)

            with tf.Session(graph=self.pred_graph) as sess:
                nn.saver.restore(sess, self.checkpoint_location)

                while not self.coord.should_stop():
                    correct_output = position_history[-1]
                    X, Y = get_io_data(position_history)
                    pred_output = sess.run(nn.predicted_outputs, { nn.inputs: X })[-1]

                    xerr = abs(correct_output[0,0]-pred_output[0,0])
                    yerr = abs(correct_output[0,1]-pred_output[0,1])

                    xerr_lst.append(xerr)
                    yerr_lst.append(yerr)

                    interval = 1
                    if ( float("{0:.1f}".format(t)) % interval < 1e-5 ):   # update the average error plot every so often
                        yerr_avg = np.mean(yerr_lst)
                        xerr_avg = np.mean(xerr_lst)

                        yerr_lst=[]  # reset the error lists to reset the averaging
                        xerr_lst=[]

                        # Dynamically update the plot
                        plt.scatter(t, xerr_avg, color='red')
                        plt.scatter(t, yerr_avg, color='blue')
                        plt.pause(1e-9)

                    if self.update_ready:
                        # Update the network parameters every so often
                        nn.saver.restore(sess, self.checkpoint_location)
                        self.update_ready = False

                    self.rate.sleep()

                    t += TIMESTEP  # update the time for plotting

        except rospy.ROSInterruptException:
            self.coord.request_stop()

    def get_predictions(self, observations, num_samples, sess, nn):
        """
        Given a sequence of observations, use dropout to generate samples from a predicted
        future distribution

        Returns:
            a (num_samples x output_size) np array of outputs
        """
        outputs = []

        for i in range(num_samples):
            # This gives a (num_steps x batch_size x output_size), ie (100 x 1 x 4), numpy array. 
            all_pred = sess.run(nn.predicted_outputs, { nn.inputs: observations })
            # We're really only interested in the last prediction: the one for the next step
            next_pred = all_pred[-1][0]   # [deltax, deltay, xdot1, ydot1] 
            
            outputs.append(next_pred)

        return np.asarray(outputs)

    
    def plot_predictions(self, x, y, observations, sess, nn, num_samples=50, num_steps=3, num_branches=10):
        """
        Plot future distributions of likely future positions

        Parameters:
            x, y         :  current position of the obstacle in cartesian coordinates
            observations :  past observations of the obstacle (changes in position)
            sess         :  an open tensorflow session with a trained model loaded
            nn           :  an LSTMNetwork instance
            num_samples  :  the number of passes through the RNN to use to estimate the distribution of future states
            num_steps    :  how many timesteps into the future to predict
            num_branches :  number of samples from the distribution to use to propagate into the future
        """

        # Get predictions for the first step 
        predictions = self.get_predictions(observations, num_samples, sess, nn)

        for i in range(num_steps):

            # Estimate the underlying distribution (with sample mean and covariance)
            mu = np.mean(predictions, axis=0)
            sigma = np.cov(predictions.T)
            
            deltax = predictions[:,0]  # use actual output of network
            deltay = predictions[:,1]

            # Update the plot - note that this takes a long time when there are too many points!
            #deltax, deltay, dx, dy = np.random.multivariate_normal(mu, sigma, 100).T  # get a bunch of sample predictions
            plt.scatter(x + deltax, y + deltay, color="blue", alpha=0.2, edgecolors="none")

            # Update x and y for the next step using the mean
            x += mu[0]
            y += mu[1]

            # Get predictions for the next step
            predictions = None
            for j in range(num_branches):
                sample_point = np.random.multivariate_normal(mu, sigma, 1)[0]
                new_obs = np.append(observations[1:], [[ sample_point ]], axis=0)

                new_pred = self.get_predictions(new_obs, num_samples/num_branches, sess, nn)  # num_samples/num_branches keeps the total number of predictions to about num_samples
                if j == 0:
                    predictions = new_pred   # initialize an np array in the first step
                else:
                    predictions = np.vstack((predictions, new_pred))

    def predict(self):
        """
        Predict the next position based on the observed history.
        """
        try:
            # Set up our graph
            nn = LSTMNetwork(self.pred_graph, self.pred_device)

            # plot parameters
            plt.xlabel("x Position")
            plt.ylabel("y Position")

            with tf.Session(graph=self.pred_graph) as sess:
                nn.saver.restore(sess, self.checkpoint_location)

                while not self.coord.should_stop():
                    x = last_x
                    y = last_y

                    # plot actual location
                    plt.scatter(x, y, color="red")

                    # plot predicted location
                    self.plot_predictions(x, y, position_history, sess, nn,
                            num_samples=10, 
                            num_steps=4, 
                            num_branches=2)

                    plt.pause(1e-5)   # this updates the plot in real time

                    # Update network parameters
                    if self.update_ready:
                        nn.saver.restore(sess, self.checkpoint_location)
                        self.update_ready = False

                    self.rate.sleep()

            plt.show()
        
        except:
            traceback.print_exc()
            self.coord.request_stop()

    def train(self):
        """
        Train the network repeatedly, using the given
        number of epochs.
        """
        try:
            NUM_EPOCHS = 100

            # Set up our graph
            nn = LSTMNetwork(self.train_graph, self.train_device)

            with tf.Session(graph=self.train_graph) as sess:
                sess.run(nn.init)
                while not self.coord.should_stop():
                    # Get input/output data from our position history
                    X, Y = get_io_data(position_history)

                    for epoch in range(NUM_EPOCHS):
                        total_error = 0
                        # train_fn triggers backprop
                        total_error += sess.run([nn.error, nn.train_fn], { nn.inputs: X, nn.outputs: Y})[0]

                    total_error /= NUM_EPOCHS

                    # Save a checkpoint
                    nn.saver.save(sess, self.checkpoint_location)
                    print("CP saved. Train error: %.6f" % (total_error))
                    self.update_ready = True   # signal that we're ready to use this new data

        except:
            traceback.print_exc()
            self.coord.request_stop()

if __name__=="__main__":
    coord = tf.train.Coordinator()
    try:
        rospy.init_node('rnn_data_collector')
        odom = rospy.Subscriber('/robot_0/base_pose_ground_truth', Odometry, odom_callback)
        rospy.sleep(1)  # wait a second for things to initialize

        ont = OnlineNetworkTrainer(coord)
        train = threading.Thread(target=ont.train)
        predict = threading.Thread(target=ont.predict)

        train.start()
        predict.start()

        coord.join([train, predict])

    except rospy.ROSInterruptException:
        pass
