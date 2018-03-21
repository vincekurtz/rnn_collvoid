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
from data_container import DataContainer
from network_variables import *
from test_trained_model import *  # includes tools for visualizing the results

def train(datafile, plot_test=True):
    """
    Run the Training Loop!

    Assues that global network variables (as in network_variables.py) are defined
    """

    NUM_EPOCHS = 20
    ITERATIONS_PER_EPOCH = 10
    NUM_STEPS = 100
    BATCH_SIZE = 2

    mydata = DataContainer(datafile)

    with tf.Session() as session:
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

if __name__=="__main__":
    datafile = "/home/vjkurtz/catkin_ws/src/rnn_collvoid/data/test_data.csv"
    train(datafile, plot_test=True)
