#!/usr/bin/env python

##
#
# Construct and save a Markov Chain model for verification with
# PRISM, Vesta, or other probabilistic model checking software
#
##

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

class PredictionTracker:
    def __init__(self, num_steps=10):
        rospy.init_node("prediction_tracker")

        self.actual_positions = []
        self.curr_pred = {}
        self.predicted_positions = []

        # track timed callback
        r = 10  # Hz
        self.dt = 1. / r
        self.last_time = rospy.get_time()

        # wait for a prediction message before setting up callbacks
        rospy.wait_for_message("/robot_0/predicted_pose/step_1", PoseWithCovarianceStamped)

        # Subscribers
        rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, self.actual_pose_cb)
        for i in range(num_steps):
            topic = "/robot_0/predicted_pose/step_%s" % (i+1)
            rospy.Subscriber(topic, PoseWithCovarianceStamped, self.predicted_pose_cb, (i,))
        

    def actual_pose_cb(self, data):

        if (rospy.get_time() > self.last_time + self.dt):
            x = data.pose.pose.position.x
            y = data.pose.pose.position.y
            t = data.header.stamp.to_sec()

            self.actual_positions.append(np.array([x,y]))
            
            # update predicted positions as well
            self.predicted_positions.append(self.curr_pred)

            self.last_time = rospy.get_time()

    def predicted_pose_cb(self, data, args):

        step_index = args[0]

        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        mu = np.array([x,y])
       
        cov = data.pose.covariance
        Sigma = np.zeros((2,2))
        Sigma[0,0] = cov[0]
        Sigma[0,1] = cov[1]
        Sigma[1,0] = cov[6]
        Sigma[1,1] = cov[7]

        self.curr_pred[step_index] = {"mu" : mu, "Sigma" : Sigma}

def prediction_errors(predicted_positions, actual_positions):
    for i in range(len(predicted_positions)-1):
        p_act = actual_positions[i+1]
        p_pred = predicted_positions[i]

        e = min_error(p_pred, p_act)
        print(e)

def min_error(predicted, actual):
    """
    Given an actual (x,y) position and a set of predictions
    {1: (mu,Sigma), 2: (mu,Sigma), ... }, return the error between the
    best prediction and the actual position.

    This handles the case that the predictions and positions that we 
    record may not be quite properly aligned, so this should identify 
    the prediction that corresponds to the given actual position.
    """
    best_step = None
    best_error = 1e6  # high number to start
    
    tolerance = 0     # anticipated error, calculated from covariance

    for step in predicted:
        error = np.linalg.norm(predicted[step]["mu"]-actual)

        if error < best_error:
            best_step = step
            best_error = error
            tolerance = np.trace(predicted[step]["Sigma"])*2e3

    return(best_error, tolerance)

def save_mc_model(predicted, actual, filename):

    num_nodes = len(predicted)-1

    mc_string = "dtmc\n\nmodule dynamic_obstacle_model\n\n"

    mc_string += "\n    s : [0..%d] init 0;\n" % num_nodes  # states 
    mc_string += "    l : [0..1] init 0;\n\n"               # labels, 1 --> "bad"

    for i in range(num_nodes):
        s = i
        sprime = i+1

        prob = 1
       
        # Get the actual error and anticipated approximate error, check how well these match
        error, tolerance = min_error(predicted[i], actual[i+1])
        label = 0 if error < tolerance else 1

        if (label == 1): print(error-tolerance)

        mc_string += "    [] s=%s -> %s : (s'=%s) & (l'=%s);\n" % (s, prob, sprime, label)

    mc_string += "\n\nendmodule"

    with open(filename, 'w') as out:
        out.write(mc_string)



if __name__=="__main__":
    p = PredictionTracker()

    rospy.sleep(5)

    actual = p.actual_positions
    predicted = p.predicted_positions

    save_mc_model(predicted, actual, "test.pm")


    
