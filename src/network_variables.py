##
# 
#  Define global variables for our LSTM network
#  Network archetecture adapted from Adapted from https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23
#
##

import tensorflow as tf
import numpy as np

# Define all global variables that make up our network structure

INPUT_SIZE = 2   # current 2D velocities
OUTPUT_SIZE = 4   # Next 2D velocities 2D change in position
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
	variational_recurrent=False,
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
