#!/usr/bin/env python

##
#
# Calculate approximate entropy of future predictions
#
##

import rospy
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseWithCovarianceStamped

class EntropyCalculator:
    """
    Subscribe to ~/predicted_pose/* and
    predict approximate entropies of corresponding
    predictions
    """
    def __init__(self, obstacle_name, num_steps=10):
        rospy.init_node("entropy_approx")
    
        self.predictions = {}

        prediction_subscribers = {}
        for i in range(num_steps):
            topic = obstacle_name +'/predicted_pose/step_%s' % (i+1)  # one indexing
            prediction_subscribers[i] = rospy.Subscriber(topic, PoseWithCovarianceStamped, self.predict_callback, (i,))

            # predictions[i] will be set in the predictor callback
            self.predictions[i] = None  
        
    def predict_callback(self, data, args):
        """
        Store mean and covariance information from 
        the subscribed prediction topics.
        """
        step_index = args[0]  # unpack from single element tuple
        
        mu = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        Sigma = np.array([[data.pose.covariance[0], data.pose.covariance[1]],
                          [data.pose.covariance[6], data.pose.covariance[7]]])

        # time this prediction is for
        t = data.header.stamp

        self.predictions[step_index] = {"mu" : mu, "Sigma" : Sigma, "time" : t}

    def mean_predictions(self, axis):
        """
        Return the series of mean predictions for the given (x or y) axis
        """
        if self.predictions[0] is None:
            return None

        return np.array([self.predictions[i]['mu'][axis] for i in self.predictions])

    def sample_entropy(self, U, m, r):
        """
        Calculate Sample Entropy
        """
        def _dist(x_i, x_j):
            return max([abs(ua -va) for ua, va in zip(x_i, x_j)])

        def _phi(m):
            x = [[U[j] for j in range(i,i+m)] for i in range(N - m+1)]
            C = [len([1 for j in range(len(x)) if i!=j and _dist(x[i],x[j]) <= r]) for i in range(len(x))]
            return float(sum(C))

        N = len(U)

        return -np.log(_phi(m+1)/_phi(m))

    def current_sample_entropy(self):
        """
        Calculate the sample entropy of the current predictions
        """
        if self.predictions[0] is None:
            return None

        pred_x = self.mean_predictions(0)
        e_x = self.sample_entropy(pred_x, 3, 0.5)
        print(e_x)

        return e_x



if __name__=="__main__":
    ec = EntropyCalculator("robot_0")
   
    se_lst = []
    try:
        while not rospy.is_shutdown():
            se = ec.current_sample_entropy()
            if se is not None:
                se_lst.append(se)

            rospy.sleep(0.1)
    except rospy.ROSInterruptException:
        print(se_lst)
        plt.plot(se_lst)
        plt.show()


