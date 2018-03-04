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

####################
# Helper Functions #
####################

class DataContainer:
    def __init__(self, datafile, train_ratio=0.8):
        
        self.t_ratio = train_ratio
        self.X = None
        self.Y = None
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        
        self.N = None # number of samples
        self.train_index = 0  # index of next unused samples
        self.test_index = 0  # index of next unused samples
        
        # Load input/output data from file
        self.load_data(datafile)

        # Generate train/test set split
        self.train_test_split()

    def reset(self):
        """Reset the counter so we can start reusing data"""
        self.train_index = 0
        self.test_index = 0

    def load_data(self, filename):
        """
        Train/test data is stored in CSV files with the following format:
            - The first few lines starting with HDR are headers
            - Subsequent lines consist of [x,y,xdot,ydot] measurements

        Return input-output data in two numpy arrays of the following format:
            Input: [[xdot0, ydot0], ...]
            Output [[x1-x0, y1-y0], ...]
        """
        raw_data = []
        myinput = []
        myoutput = []

        # load raw data from CSV
        with open(filename,'r') as incsv:
            reader = csv.reader(incsv)

            for row in reader:
                if row[0][0] == "H":
                    # ignore headers
                    pass
                else:
                    raw_data.append(row)

        # Create a input/output lists
        for i in range(len(raw_data)-1):   # traverse in order, only up to N-1
            x0 = float(raw_data[i][0])
            y0 = float(raw_data[i][1])
            xdot0 = float(raw_data[i][2])
            ydot0 = float(raw_data[i][3])
            x1 = float(raw_data[i+1][0])
            y1 = float(raw_data[i+1][1])

            myinput.append([xdot0, ydot0])
            myoutput.append([float("{0:.4f}".format(x1-x0)), float("{0:.4f}".format(y1-y0))])  # limit to 4 decimals

        # Converty to np arrays and store
        self.X = np.asarray(myinput)
        self.Y = np.asarray(myoutput)
        self.N = len(self.X)

    def train_test_split(self):
        """
        Divide the given dataset into a training set and a testing
        set, with the given ratio. I.e. by default 80 percent goes
        to training and 20 percent to testing. 
        """
        
        self.N = len(self.X)
        cutoff = int(self.N*self.t_ratio)

        self.train_X = self.X[0:cutoff]
        self.train_Y = self.Y[0:cutoff]

        self.test_X = self.X[cutoff:]
        self.test_Y = self.Y[cutoff:]
    
    def get_next_batch(self, num_steps, batch_size, dataset="Train"):
        """
        Give a set of training data and a starting index, generate a batch as follows:

        x: np.array
            2D velocity in cartesian space
        y: np.array
            Subsequent change in position at the next time step
        """
        x = np.empty((num_steps, batch_size, 2))
        y = np.empty((num_steps, batch_size, 2))

        # Determine which dataset to use
        if dataset == "Train":
            input_data = self.train_X
            output_data = self.train_Y
            index = self.train_index
        elif dataset == "Test":
            input_data = self.test_X
            output_data = self.test_Y
            index = self.test_index
        else:
            raise AssertionError("invalid dataset '%s'" % dataset)
     
        # Ensure there is enough data
        max_index = index + batch_size*num_steps
        if max_index > len(input_data):
            raise AssertionError("Not enough data! You asked for %s new data points, but I only have %s left" % (batch_size*num_steps, len(input_data)-index))

        for i in range(batch_size):
            xdot = input_data[index:index+num_steps][:,0]
            ydot = input_data[index:index+num_steps][:,1]
            dx = output_data[index:index+num_steps][:,0]
            dy = output_data[index:index+num_steps][:,1]
            x[:, i, 0] = xdot
            x[:, i, 1] = ydot
            y[:, i, 0] = dx
            y[:, i, 1] = dy

            index += num_steps

        # Update the index for next time
        if dataset == "Train":
            self.train_index = index
        else:
            self.test_index = index

        return x, y


##################################################################################
# Set up LSTM Network                                                            # 
# Adapted from https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23 #
##################################################################################

INPUT_SIZE = 2   # x and y velocities
OUTPUT_SIZE = 2   # x and y positions
RNN_HIDDEN = 20 
TINY = 1e-6
LEARNING_RATE = 0.01

inputs = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))

# Create a basic LSTM cell, there are other options too
cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)

# Add dropout
cell = tf.nn.rnn_cell.DropoutWrapper(
        cell,
        input_keep_prob=0.8,
        output_keep_prob=0.8,
        state_keep_prob=0.9,
        variational_recurrent=True,   # apply same dropout mask every step, as per https://arxiv.org/abs/1512.05287 
        input_size=INPUT_SIZE,
        dtype=tf.float32,
        seed=None,
        dropout_state_filter_visitor=None
)

# Create initial state as all zeros
batch_size = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)

# Given a set of inputs, return a tuple with rnn outputs and rnn state
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

# Project rnn outputs to our OUTPUT_SIZE
final_projection = lambda x: tf.contrib.layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=None)
predicted_outputs = tf.map_fn(final_projection, rnn_outputs)

# Compute the error that we want to minimize
error = tf.losses.huber_loss(outputs, predicted_outputs)
#error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
#error = tf.reduce_mean(error)

# Optimization
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

# Accuracy measurment
accuracy = tf.reduce_mean(tf.abs(outputs - predicted_outputs))

###########################
#  Run the Training Loop! #
###########################

NUM_EPOCHS = 50
ITERATIONS_PER_EPOCH = 2
NUM_STEPS = 100
BATCH_SIZE = 2


datafile = "/home/vjkurtz/catkin_ws/src/rnn_collvoid/data/test_data.csv"
mydata = DataContainer(datafile)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epoch in range(NUM_EPOCHS):
    epoch_error = 0
    mydata.reset()   # retrain on the same data just for now, for testing purposes
    for _ in range(ITERATIONS_PER_EPOCH):
        # Get training data for the next batch
        x, y = mydata.get_next_batch(NUM_STEPS, BATCH_SIZE)

        # train_fn triggers backprop
        epoch_error += session.run([error, train_fn], { inputs: x, outputs: y})[0]

    epoch_error /= ITERATIONS_PER_EPOCH
    print("Epoch %d, train error: %.4f" % (epoch, epoch_error))

# Test on the test set!
test_x, test_y = mydata.get_next_batch(num_steps=NUM_STEPS, batch_size=1, dataset="Test")
print("")
predict1 = session.run(predicted_outputs, { inputs: test_x, outputs: test_y })
predict2 = session.run(predicted_outputs, { inputs: test_x, outputs: test_y })

for i in range(len(predict1)):
    print("%s   ---->    %s    ||    %s" % (test_y[i][0], predict1[i][0], predict2[i][0]))

session.close()

