#!/usr/bin/env python

##
#
# Just a quick script to plot some stuff
#
##

import matplotlib.pyplot as plt

# our (threshold) controller
num_steps = [2,3,4,5,6,7]
total_time = [ 3.4, 4.2, 7.5, 22.5, 194.4, 3379.1 ]
collision_prob = [ 0.00456, 0.0316, 0.0117859, 0.065516, 0.0632168, 0.0675338 ]

plt.subplot(2, 1, 1)
plt.plot(num_steps, collision_prob, 'ro-')
plt.title("Our Controller")
plt.ylabel("Collision Probability")
plt.ylim(0,0.15)

plt.subplot(2, 1, 2)
plt.plot(num_steps, total_time, 'o-')
plt.ylabel("Processing Time (s)")
plt.xlabel("Timesteps")
plt.show()

# naive (always down) controller
n_num_steps = [2,3,4,5,6,7]
n_total_time = [3.25, 3.374, 3.88, 7.941, 86.7, 3144.14]
n_collision_prob = [0, 0.003636, 0.003506, 0.03721, 0.040573, 0.1338]

plt.subplot(2, 1, 1)
plt.plot(n_num_steps, n_collision_prob, 'ro-')
plt.title("Naive Controller (always move down)")
plt.ylabel("Collision Probability")
plt.ylim(0,0.15)

plt.subplot(2, 1, 2)
plt.plot(n_num_steps, n_total_time, 'o-')
plt.ylabel("Processing Time (s)")
plt.xlabel("Timesteps")

plt.show()

