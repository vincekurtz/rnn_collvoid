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
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

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

def plot_trajectory(traj):
    """
    Make a matplotlib plot of the path of the robot
    given a list of changes in position (could be predicted or actual)
    """
    # Set up axes
    figs, axes = plt.subplots(2,1)
    (ax1, ax2) = axes

    radius = .2
    x = 0  # define initial position at the origin
    y = 0
    patches = []
    xs = []
    ys = []
    t = 0
    time = [t + 1 for i in range(len(traj))]
    print(len(time))

    # Unpack the data
    for delta in traj:
        x = x + delta[0][0]
        y = y + delta[0][1]
        circle = Circle((x,y), radius)
        patches.append(circle)
        xs.append(x)
        ys.append(y)

    # Plot state space trajectory
    p = PatchCollection(patches, alpha=0.4)
    ax1.add_collection(p)
    ax1.autoscale_view()
    ax1.set_title("State Space Trajectory")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Plot x, y positions vs time
    ax2.plot(xs)
    ax2.plot(ys)
    ax2.set_title("Position vs Time")

    plt.show()

def plot_comparison(predicted_trajectories, actual_trajectory):
    """
    Plot multiple predicted trajectories (i.e. those generated with
    dropout) and one actual trajectory.

    Assumes that actual_trajectory (and each element of predicted_trajectories)
    were created using a batch size of one. 
    """
    N = len(predicted_trajectories)  # the number of different predictions
 
    # Set up multiple axes
    figs, axes = plt.subplots(2,1)
    (ax1, ax2) = axes
    ax1.set_title("X position change")
    ax2.set_title("Y position change")

    actual_x = actual_trajectory[:,0,0]
    actual_y = actual_trajectory[:,0,1]

    for i in range(N):
        pred_x = predicted_trajectories[i][:,0,0]
        pred_y = predicted_trajectories[i][:,0,1]
        ax1.plot(pred_x, color="b", alpha=0.5)
        ax2.plot(pred_y, color="b", alpha=0.5)

    ax1.plot(actual_x, color="r")
    ax1.set_xlabel("timestep")
    ax1.set_ylabel("position change")
    ax2.plot(actual_y, color="r")
    ax2.set_xlabel("timestep")
    ax2.set_ylabel("position change")
    plt.show()


def train(datafile, plot_test=True):
    ##################################################################################
    # Set up LSTM Network                                                            # 
    # Adapted from https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23 #
    ##################################################################################

    INPUT_SIZE = 2   # x and y velocities
    OUTPUT_SIZE = 2   # x and y positions
    RNN_HIDDEN = 256 
    LEARNING_RATE = 0.01

    inputs = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))
    outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))

    # Create a basic LSTM cell, there are other options too
    cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)

    # Add dropout
    cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=0.9,
            output_keep_prob=0.9,
            state_keep_prob=1.0,
            variational_recurrent=False,   # apply same dropout mask every step, as per https://arxiv.org/abs/1512.05287 
                                           # seems to lead to strange behavior where 0 is predicted all the time
            input_size=INPUT_SIZE,
            dtype=tf.float32,
            seed=None
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

    # Optimization
    train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

    # Accuracy measurment
    accuracy = tf.reduce_mean(tf.abs(outputs - predicted_outputs))

    ###########################
    #  Run the Training Loop! #
    ###########################

    NUM_EPOCHS = 20
    ITERATIONS_PER_EPOCH = 10
    NUM_STEPS = 100
    BATCH_SIZE = 2

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
        print("Epoch %d, train error: %.6f" % (epoch, epoch_error))

    # Save the trained model
    save_name = "/home/vjkurtz/catkin_ws/src/rnn_collvoid/tmp/LSTM_saved_model"
    saver = tf.train.Saver()
    saver.save(session, save_name)

    if plot_test:
        # Test on the test set!
        test_x, test_y = mydata.get_next_batch(num_steps=NUM_STEPS, batch_size=1, dataset="Test")

        # Use dropout to get a bunch of predictions
        predicts = []
        for i in range(100):
            p = session.run(predicted_outputs, { inputs: test_x, outputs: test_y })
            predicts.append(p)

        plot_comparison(predicts, test_y)
    
    session.close()

if __name__=="__main__":
    datafile = "/home/vjkurtz/catkin_ws/src/rnn_collvoid/data/test_data.csv"
    train(datafile)
