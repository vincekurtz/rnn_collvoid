#!/usr/bin/env python

##
#
# Train a simple LSTM RNN to predict future
# positions based on current position and velocity. 
# 
# Then verify these predictions using statistical model checking. 
# Assumes that position information is published to /last_position_change
# at a moderate and regular rate.
#
##

import rospy
import threading
import sys 
import numpy as np
import tensorflow as tf
import matplotlib.patches as patches
from geometry_msgs.msg import Vector3

class LSTMNetwork():
    """
    A representation of an LSTM network, using the 
    specified graph
    """
    def __init__(self, graph, device, steps=10, variational_recurrent=False):

        self.num_steps=steps

        with graph.as_default():
            with graph.device(device):
                INPUT_SIZE = 2                   # last changes in position
                OUTPUT_SIZE = 2*self.num_steps   # Next N changes in position
                RNN_HIDDEN = 20
                LEARNING_RATE = 0.003

                inputs = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))
                outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))

                # Create a basic LSTM cell, there are other options too
                cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
                #cell = tf.nn.rnn_cell.BasicRNNCell(RNN_HIDDEN)

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
    def __init__(self, steps=4):
        # List of previous actual positions
        self.position_history = None

        # List of previous predicted positions
        self.predicted_pos = []

        # The coordinator will tell separate threads when to stop
        self.coord = tf.train.Coordinator()

        # Set up two tensorflow graphs for training and prediction
        self.train_graph = tf.Graph()
        self.pred_graph = tf.Graph()
        
        self.train_device = '/cpu:0'
        self.pred_device = '/cpu:0'

        # Specify where to save checkpoints
        self.checkpoint_location = "/tmp/statmc_graph.checkpoint"

        # Keep track of when a new update to the model is ready
        self.update_ready = False
        self.predict_ready = False

        # Number of steps into the future to predict
        self.num_steps = steps
       
        # ROS parameters 
        rospy.init_node('rnn_online_predictor')

        # keep track of number of steps predicted
        self.i = 0

        self.start_online_train_predict()

    def start_online_train_predict(self):
        """
        Start performing online training in another thread
        """
        print("\n===> Beginning New Trial\n")

        predict = threading.Thread(target=self.start_predicting)
        train = threading.Thread(target=self.train)
        
        train.start()
        predict.start()

        self.coord.join([predict, train])

    def odom_callback(self, data, args):
        """
        Updates actual data in self.position_history, triggers a new prediction,
        and evaluates past predictions based on the latest data. 
        """
        sess = args[0]
        nn = args[1]

        N = 20  # Length of a given run
        
        # update training data in self.position_history
        if self.position_history is not None:
            self.position_history = np.vstack((self.position_history, np.array([[[data.x, data.y]]])))
        else:
            self.position_history = np.array([[[data.x, data.y]]])

        # Make and evaluate predictions
        if self.predict_ready:
            # Make a prediction using the latest data
            self.predict(sess, nn)

            # Evaluate previous predictions
            sat = self.eval_predictions()
            
            # Use warning level so we can see what's going on while grepping stdout
            rospy.logwarn("Evaluating predictions ... %s / %s" % (self.i, N))

            # Stop if any prediction fails to satisfy the specification
            if not sat:
                rospy.logwarn("Failed to satisfy specification!")
                print("\n===> Failed to satisfy specification!")
                print("===> Result: 0\n")
                self.coord.request_stop()
                rospy.signal_shutdown("Failed to satisfy specification")

            # Stop after a fixed number of iterations
            self.i += 1
            if self.i > N:
                rospy.logwarn("Specification satisfied!")
                print("\n===> Specification satisfied!")
                print("===> Result: 1\n")
                self.coord.request_stop()
                rospy.signal_shutdown("Reached target iterations")
        else:
            print("Collecting training data ... %s / %s " % (len(self.position_history), self.num_steps+1) )
            rospy.logwarn("Collecting training data ... %s / %s " % (len(self.position_history), self.num_steps+1) )
    
    def eval_predictions(self):
        """
        Evaluate the given predictions against the actual historical data. Specifically,
        test to make sure the error for all predicted future steps is below a threshold
        defined by the covariance of that prediction. 
        """

        # Align predictions such that predicted[i] corresponds to predictions made
        # when the actual position was actual[i]
        actual = self.position_history[self.num_steps+1:]
        predicted = self.predicted_pos

        # We need to have actual data for self.num_steps after any prediction
        if len(actual) < self.num_steps + 1:
            return True

        # find the index of the latest evaluatable prediction 
        idx = len(actual)-1 - self.num_steps

        for i in range(self.num_steps):
            a = actual[idx+i+1]    # actual result for this step
            p = predicted[idx][i]  # latest prediction for this step

            # scale up sigma, since this is easier than setting an insane threshold directly
            sat = self.within_threshold(a[0], p["mu"], p["Sigma"]*1e3, 0.99)

            if not sat:
                return False
        return True


    def within_threshold(self, val, mu, Sigma, thresh):
        """
        Indicate if 'val' is within the given probabilistic threshold of
        the 2d Gaussian distribution described by mu, Sigma
        """
        # Construct an ellipse containing all values within the threshold
        
        # Maximum safe Mahalanobis distance from mu
        safe_m_distance = np.sqrt(-2*np.log(1-thresh))

        # Eigenvalues and vectors of the covariance
        (W,v) = np.linalg.eig(Sigma)

        # Ellipse parameters
        angle = np.angle((v[0,0]+v[1,0]*1j),deg=True)
        width = safe_m_distance*np.sqrt(W[0])
        height = safe_m_distance*np.sqrt(W[1])
        center = (mu[0], mu[1])

        ellipse = patches.Ellipse(center, width, height, angle=angle)
        result = ellipse.contains_point(val)

        if result == False:
            print("")
            print("val: %s" % val)
            print("center: %s\nwidth: %s\nheight: %s\nangle: %s" % (center, width, height, angle))
        
        return(result)

    def get_io_data(self):
        """
        Given a history of delta_x and delta_y's, return input
        and output data that can be used to train the network.
        """
        if (self.position_history is None) or (len(self.position_history) <= self.num_steps):
            # Wait until we have enough data for at least one step
            return None

        N = len(self.position_history)
        
        # use the whole history as input
        ipt = self.position_history[0:N-self.num_steps]

        # Calculate outputs as the subsequent steps
        o = [ self.position_history[i:N-(self.num_steps-i)] for i in range(1,self.num_steps+1)]
        opt = np.concatenate(o, axis=2)

        # Add gaussian measurement noise
        ipt = ipt + 0.001*np.random.normal(size=ipt.shape)
        opt = opt + 0.001*np.random.normal(size=opt.shape)

        return(ipt, opt)
    
    def start_predicting(self):
        """
        Initialize the callback so we start making predictions
        """
        try:
            # Network and session for predicting
            with tf.Session(graph=self.pred_graph) as sess:
                nn = LSTMNetwork(self.pred_graph, self.pred_device, steps=self.num_steps)
                sess.run(nn.init)

                # Start subscribing to /last_position_change
                # We start here so that the prediction network can be an argument to the callback
                self.pred_sub = rospy.Subscriber('/last_position_change', Vector3, self.odom_callback, callback_args=(sess,nn))
                
                rospy.spin()

        except rospy.ROSInterruptException:
            print("prediction_request_stop")
            self.coord.request_stop()

    def predict(self, sess, nn, num_samples=10):
        """
        Predict the next positions based on the observed history.
        """
            
        if self.update_ready:
            nn.saver.restore(sess, self.checkpoint_location)
            self.update_ready = False

        all_pred = self.get_raw_predictions(sess, nn, num_samples)

        p = {}

        for i in range(self.num_steps):
            
            # Get predictions for this timestep
            predictions = all_pred[:,2*i:2*i+2]

            # Make Gaussian MLE of underlying distribution
            mu = np.mean(predictions, axis=0)
            Sigma = np.cov(predictions.T)
            
            # Update prediction data for this step
            p[i] = {"mu": mu, "Sigma": Sigma}
        
        # Update all prediction data
        self.predicted_pos.append(p)

    def get_raw_predictions(self, sess, nn, num_samples):
        """
        Use dropout to generate samples from a predicted
        future distribution
        """
        outputs = []

        for i in range(num_samples):
            # This gives a (num_steps x batch_size x output_size), ie (100 x 1 x 4), numpy array. 

            # Add noise to input
            ipt = self.position_history + 0.001*np.random.normal(size=self.position_history.shape)

            all_pred = sess.run(nn.predicted_outputs, { nn.inputs: ipt })

            # We're really only interested in the last prediction: the one for the next step
            next_pred = all_pred[-1][0]

            outputs.append(next_pred)

        return np.asarray(outputs)

    def train(self):
        try:

            # Set up our graph
            nn = LSTMNetwork(self.train_graph, self.train_device, steps=self.num_steps)

            with tf.Session(graph=self.train_graph) as sess:
                sess.run(nn.init)
                while not self.coord.should_stop():
                    self.train_once(nn,sess)

        except rospy.ROSInterruptException:
            print("caught in train loop")
            self.coord.request_stop()

    def train_once(self, nn, sess, NUM_EPOCHS=100):
        # Get input/output data from our position history
        io_data = self.get_io_data()

        # Don't do anything if we can't get enough input-output data
        if io_data is None:
            return None
        
        X,Y = io_data

        for epoch in range(NUM_EPOCHS):
            total_error = 0
            # train_fn triggers backprop
            total_error += sess.run([nn.error, nn.train_fn], { nn.inputs: X, nn.outputs: Y})[0]

        total_error /= NUM_EPOCHS

        # Save a checkpoint
        nn.saver.save(sess, self.checkpoint_location)
        
        # signal that we're ready to use this new data
        self.update_ready = True   
        self.predict_ready = True



if __name__=="__main__":
    try:
        nn = OnlinePredictionNetwork(steps=10)

    except rospy.ROSInterruptException:
        print("caught in final exception")
        nn.coord.request_stop()
