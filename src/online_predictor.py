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
from geometry_msgs.msg import PoseWithCovarianceStamped
import threading
import traceback

import matplotlib.pyplot as plt

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


class OnlinePredictionNetwork():
    """
    A recurrent network for predicting obstacle motion based on online 
    observations. Uses two copies of a network in parallel: one for
    training and one for predicting.
    """
    def __init__(self, robot_name, steps=4):
        # TRAINING DATA: these variables are updated in real time
        self.last_x = None
        self.last_y = None
        self.position_history = None

        self.last_time = None  # so we only keep track of data so often

        # The coordinator will tell separate threads when to stop
        self.coord = tf.train.Coordinator()

        # Set up two tensorflow graphs for training and prediction
        self.train_graph = tf.Graph()
        self.pred_graph = tf.Graph()
        
        self.train_device = '/cpu:0'
        self.pred_device = '/cpu:0'

        # Specify where to save checkpoints
        self.checkpoint_location = "/tmp/graph.checkpoint"

        # Keep track of when a new update to the model is ready
        self.update_ready = False

        # Number of steps into the future to predict
        self.num_steps = steps
       
        # ROS parameters 
        self.TIMESTEP = 0.1   # seconds between samples
        self.header = None    # for keeping track of the time
        rospy.init_node('rnn_online_predictor')
        odom = rospy.Subscriber(robot_name + '/base_pose_ground_truth', Odometry, self.odom_callback)

        self.prediction_publishers = {}   # a dictionary of publishers so we can publish a PoseWithCovarianceStamped
                                          # for each timestep we predict into the future
        for i in range(self.num_steps):
            prediction_pub = rospy.Publisher(robot_name + '/predicted_pose/step_%s' % (i+1), PoseWithCovarianceStamped, queue_size=20)
            self.prediction_publishers[i] = prediction_pub

        rospy.sleep(1)  # wait a second for things to initialize
        self.rate = rospy.Rate(1.0/self.TIMESTEP)

    def start_online_prediction(self):
        """
        Start performing online training and prediction
        """
        train = threading.Thread(target=self.train)
        predict = threading.Thread(target=self.predict, kwargs={'mplot':False})

        train.start()

        # Wait until we've saved a trained graph
        while not self.update_ready:
            pass
        predict.start()

        self.coord.join([train, predict])

    def odom_callback(self, data):
        """
        Updates the stored training data in self.position_history
        and the tracked location in self.last_{x|y}
        """
        self.header = data.header   # update a global header for use in other messages too

        if self.last_time is None:
            # initialize the time counter: this must happen
            # after the node an everything is started
            self.last_time = rospy.get_time()

        if (rospy.get_time() > self.last_time + self.TIMESTEP):
            curr_x = data.pose.pose.position.x
            curr_y = data.pose.pose.position.y

            if self.last_x is not None:
                deltax = curr_x - self.last_x
                deltay = curr_y - self.last_y

                if self.position_history is not None:
                    self.position_history = np.vstack((self.position_history, np.array([[[deltax, deltay]]])))
                else:
                    self.position_history = np.array([[[deltax, deltay]]])

            self.last_x = curr_x
            self.last_y = curr_y
            self.last_time = rospy.get_time()

    def get_io_data(self, pos_hist):
        """
        Given a history of delta_x and delta_y's, return input
        and output data that can be used to train the network.
        """
        N = len(pos_hist)
        n = 100  # number of points to use
       
        if (N < n):
            # use the whole history
            ipt = pos_hist[0:N-1]
            opt = pos_hist[1:N]
        else:
            # don't bother dealing with the old bits
            ipt = pos_hist[N-n:N-1]
            opt = pos_hist[N-n+1:N]

        return(ipt, opt)

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
                    correct_output = self.position_history[-1]
                    X, Y = get_io_data(self.position_history)
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

                    t += self.TIMESTEP  # update the time for plotting

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

    
    def plot_predictions(self, x, y, observations, sess, nn, num_samples=50, num_branches=10, plot_raw=False):
        """
        Plot future distributions of likely future positions. Use a particle filtering approach
        to propagate predictions into the future. 

        Parameters:
            x, y         :  current position of the obstacle in cartesian coordinates
            observations :  past observations of the obstacle (changes in position)
            sess         :  an open tensorflow session with a trained model loaded
            nn           :  an LSTMNetwork instance
            num_samples  :  the number of passes through the RNN to use to estimate the distribution of future states
            num_branches :  number of samples from the distribution to use to propagate into the future
            plot_raw     :  whether to plot actual network output or the gaussian fit
        """

        # Get predictions for the first step 
        predictions = self.get_predictions(observations, num_samples, sess, nn)

        for i in range(self.num_steps):

            # Estimate the underlying distribution (with sample mean and covariance)
            mu = np.mean(predictions, axis=0)
            sigma = np.cov(predictions.T)
           
            if plot_raw:
                # use output of network
                deltax = predictions[:,0]
                deltay = predictions[:,1]
            else:
                # use our gaussian fit
                deltax, deltay = np.random.multivariate_normal(mu, sigma, 100).T

            # Update the plot - note that this takes a long time when there are too many points!
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

    def make_future_predictions(self, x, y, observations, sess, nn, num_samples=50, num_branches=10):
        """
        Calculate distributions of likely future positions using a particle filtering approach,
        and publish these predictions to a ROS topic. 

        Parameters:
            x, y         :  current position of the obstacle in cartesian coordinates
            observations :  past observations of the obstacle (changes in position)
            sess         :  an open tensorflow session with a trained model loaded
            nn           :  an LSTMNetwork instance
            num_samples  :  the number of passes through the RNN to use to estimate the distribution of future states
            num_branches :  number of samples from the distribution to use to propagate into the future
        """

        # Get predictions for the first step 
        predictions = self.get_predictions(observations, num_samples, sess, nn)

        # Configure the ROS message
        pose_prediction = PoseWithCovarianceStamped()

        for i in range(self.num_steps):

            # Estimate the underlying distribution (with sample mean and covariance)
            mu = np.mean(predictions, axis=0)
            sigma = np.cov(predictions.T)
           
            # Update x and y for the next step using the mean
            x += mu[0]
            y += mu[1]

            # Update the ROS pose message
            pose_prediction.header = self.header   # to match what we get from odometry
            pose_prediction.pose.pose.position.x = x
            pose_prediction.pose.pose.position.y = y
            pose_prediction.pose.covariance[0] = sigma[0,0]  # x and x
            pose_prediction.pose.covariance[1] = sigma[0,1]  # x and y
            pose_prediction.pose.covariance[6] = sigma[1,0]  # y and x
            pose_prediction.pose.covariance[7] = sigma[1,1]  # x and y

            self.prediction_publishers[i].publish(pose_prediction)

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

    def predict(self, mplot=False):
        """
        Predict the next position based on the observed history.
        """
        try:
            # Set up our graph
            nn = LSTMNetwork(self.pred_graph, self.pred_device)

            if mplot:
                # plot parameters
                plt.xlabel("x Position")
                plt.ylabel("y Position")

            with tf.Session(graph=self.pred_graph) as sess:
                nn.saver.restore(sess, self.checkpoint_location)

                while not self.coord.should_stop():
                    x = self.last_x
                    y = self.last_y

                    if mplot:
                        # plot actual location
                        plt.scatter(x, y, color="red")

                        # plot predicted location
                        self.plot_predictions(x, y, self.position_history, sess, nn,
                                num_samples=10, 
                                num_branches=2,
                                plot_raw=False)
                    
                    self.make_future_predictions(x, y, self.position_history, sess, nn,
                            num_samples=20, 
                            num_branches=5)

                    if mplot:
                        plt.pause(1e-5)   # this updates the plot in real time

                    # Update network parameters
                    if self.update_ready:
                        nn.saver.restore(sess, self.checkpoint_location)
                        self.update_ready = False

                    self.rate.sleep()

            if mplot:
                plt.show()
       
        except rospy.ROSInterruptException:
            self.coord.request_stop()
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
                    X, Y = self.get_io_data(self.position_history)

                    for epoch in range(NUM_EPOCHS):
                        total_error = 0
                        # train_fn triggers backprop
                        total_error += sess.run([nn.error, nn.train_fn], { nn.inputs: X, nn.outputs: Y})[0]

                    total_error /= NUM_EPOCHS

                    # Save a checkpoint
                    nn.saver.save(sess, self.checkpoint_location)
                    #print("CP saved. Train error: %.6f" % (total_error))
                    self.update_ready = True   # signal that we're ready to use this new data

        except rospy.ROSInterruptException:
            self.coord.request_stop()
        except:
            traceback.print_exc()
            self.coord.request_stop()

if __name__=="__main__":
    try:
        nn = OnlinePredictionNetwork('robot_0')
        nn.start_online_prediction()

    except rospy.ROSInterruptException:
        pass
