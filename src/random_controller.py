#!/usr/bin/env python

##
#
# A randomized controller for statistical model checking. 
# Publishes directly to /last_position_change so we don't
# have to actually run Stage
#
##

import rospy
import numpy as np
from geometry_msgs.msg import Vector3

# initialize the node, publisher, subscriber
rospy.init_node("random_controller", anonymous=True)
pub = rospy.Publisher('/last_position_change', Vector3, queue_size=10)

# Number of steps to use in a run
N = 100

# Publishing rate
rate = rospy.Rate(3)  # Hz

# Data to publish
p = Vector3()

# Posible changes in position that we could take
delta_ps = np.array([[1,0],[0,1],[-1,0],[0,-1]])
p_idx = np.array([i for i in range(len(delta_ps))])

for i in range(N):

    if not rospy.is_shutdown():
        # choose a random direction
        #idx = np.random.choice(p_idx)

        idx = 0
        delta_p = delta_ps[idx]

        p.x = delta_p[0]
        p.y = delta_p[1]

        pub.publish(p)

        rate.sleep()
    
