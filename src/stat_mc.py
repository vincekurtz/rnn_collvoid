#!/usr/bin/env python

##
#
# Contains several methods for controlling a robot based on 
# predicted future positions
#
##

import rospy
import sys
import numpy as np
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int8
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.patches as patches

class DynamicCAController():
    """
    A dynamic collision avoidance controller that 
    uses predictions of obstacle behavior to output
    safe velocities for another robot in the workspace.
    """
    def __init__(self, robot_name, obstacle_name, steps=10, theta=0.1):
        # Number of steps into the future that are predicted
        self.num_steps = steps

        # My location and preferred velocity
        self.x = None
        self.y = None
        self.cmd_vel_nominal = Twist()

        # maximum allowable probability of collision
        self.theta = theta

        # an object to hold predicted occupied locations
        self.predictions = {}           

        # ROS setup
        rospy.init_node("controller")
        self.control_pub = rospy.Publisher(robot_name + '/cmd_vel', Twist, queue_size=100)
        self.reset_sim = rospy.ServiceProxy('reset_positions', Empty)
        rospy.Subscriber(robot_name + '/base_pose_ground_truth', Odometry, self.odom_callback)
        rospy.Subscriber(robot_name + '/cmd_vel_nominal', Twist, self.nominal_vel_callback)
        rospy.Subscriber(robot_name + '/is_crashed', Int8, self.crash_callback)
        rospy.Subscriber(obstacle_name + '/is_crashed', Int8, self.crash_callback)
        rospy.Subscriber(robot_name + '/goal_reached', Bool, self.goal_callback)

        # Prediction subscribers: one for each timestep in the future
        prediction_subscribers = {}
        
        for i in range(self.num_steps):
            
            topic = obstacle_name + '/predicted_pose/step_%s' % (i+1)  # one indexing
            prediction_subscribers[i] = rospy.Subscriber(topic, PoseWithCovarianceStamped, self.predict_callback, (i,))

            # predictions[i] will be set in the predictor callback
            self.predictions[i] = None  

        # rate for publication of command velocity
        self.rate = rospy.Rate(10)  # Hz

        # Define a safe distance from the obstacle
        obstacle_radius = 0.125
        my_radius = 0.125
        buffer_dist = 0.6
        self.safe_radius = obstacle_radius + my_radius + buffer_dist

        # keep track of a given trial (for statistical model checking)
        self.trial_finished = False
        self.trial_success = False

    def goal_callback(self, data):
        if data.data:
            # we've reached the goal
            self.trial_finished = True
            self.trial_success = True

    def crash_callback(self, data):
        if data.data:
            # we've crashed
            self.trial_finished = True
            self.trial_success = False

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

    def odom_callback(self, data):
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y

    def nominal_vel_callback(self, data):
        self.cmd_vel_nominal = data

    def collision_probability(self, rv, position):
        """
        Returns the probability of collision at a given position,
        with rv a random variable that describes a probable future
        position of an obstacle. 

        Provides an upper bound estimate on collision probability
        for a circular obstacle by considering the 2D CDF of a
        square that encloses the safe radius around the given position
        """
        top_right = lambda x : [x[0] + self.safe_radius, x[1] + self.safe_radius]
        bottom_right = lambda x : [x[0] + self.safe_radius, x[1] - self.safe_radius]
        top_left = lambda x : [x[0] - self.safe_radius, x[1] + self.safe_radius]
        bottom_left = lambda x :[x[0] - self.safe_radius, x[1] - self.safe_radius]

        return rv.cdf(top_right(position)) - rv.cdf(bottom_right(position)) - rv.cdf(top_left(position)) + rv.cdf(bottom_left(position))

    def nominal_control(self):
        """
        Output a nominal velocity (w/o collision avoidance) to follow
        """
        v = Twist()

        # Use a proportional controller to keep the x value close to zero
        kp_x = 1

        v.linear.y = -0.7
        v.linear.x = -kp_x * self.x

        return v

    def NPVO_control(self):
        """
        A controller based on the concept of a Nonlinear Probibalistic
        Velocity Obstacle. At each timestep, a velocity is chosen that
        is as close as possible to the desired velocity, but that
        will never lead to a collision (according to our current
        predictions, with a given probability). 
        """

        # Wait until we have our position and some predictions
        while (self.x is None) or (self.predictions[0] is None):
            self.rate.sleep()

        print("Beginning control sequence")
        while not rospy.is_shutdown():

            # Get a desired velocity
            vdes = self.cmd_vel_nominal

            # Get collision avoidance constraints
            cons = self.get_constraints()

            # Set up the performance cost: minimize the difference between the chosen and 
            # desired velocities
            performance_cost = lambda x : (x[0] - vdes.linear.x)**2 + (x[1] - vdes.linear.y)**2

            res = minimize(performance_cost, [vdes.linear.x, vdes.linear.y], constraints=cons, options={"maxiter": 2000}, method="COBYLA")

            best_vel = Twist()
            best_vel.linear.x = res.x[0]
            best_vel.linear.y = res.x[1]

            # Move the robot in the chosen position
            self.control_pub.publish(best_vel)

            self.rate.sleep()

    def NPVO_verification(self):
        """
        Run our NPVO controller a bunch of times for statistical model checking
        """

        # Wait until we have our position and some predictions
        while (self.x is None) or (self.predictions[0] is None):
            self.rate.sleep()

        self.rate.sleep()

        print("Beginning control sequence")

        self.trial_finished = False
        self.trial_success = False
        
        while not self.trial_finished and not rospy.is_shutdown():

            # Get a desired velocity
            vdes = self.cmd_vel_nominal

            # Get collision avoidance constraints
            cons = self.get_constraints()

            # Set up the performance cost: minimize the difference between the chosen and 
            # desired velocities
            performance_cost = lambda x : (x[0] - vdes.linear.x)**2 + (x[1] - vdes.linear.y)**2

            res = minimize(performance_cost, [vdes.linear.x, vdes.linear.y], constraints=cons, options={"maxiter": 2000}, method="COBYLA")

            best_vel = Twist()
            best_vel.linear.x = res.x[0]
            best_vel.linear.y = res.x[1]

            # Move the robot in the chosen position
            self.control_pub.publish(best_vel)

            self.rate.sleep()

        # Record the results of the trial
        print("")
        print("TRIAL FINISHED. SUCCESS = %d" % self.trial_success)
        print("")


    def get_constraints(self):
        """
        Return a list of minimization constraints such that a chosen velocity doesn't
        lead to a collision
        """

        cons = []

        # Current time and position of the controllable robot
        my_x = self.x
        my_y = self.y 
        my_time = rospy.get_rostime()

        for step in self.predictions:
            mu = self.predictions[step]["mu"]
            Sigma = self.predictions[step]["Sigma"]
            t = self.predictions[step]["time"]

            pred = (mu, Sigma, t)

            c = {
                    'type' : 'ineq',
                    'fun' : lambda x, pred, my_x, my_y, my_time : self.distance_from_collision(pred, x, my_x, my_y, my_time) - 1,
                    'args' : (pred, my_x, my_y, my_time)
                  }

            cons.append(c)

        return cons


    def distance_from_collision(self, prediction, velocity, my_x, my_y, my_time):
        """
        For a given velocity (vx, vy), find a normalized distance (r) from
        a collision. 
        
        If r <= 1, the given velocity is unsafe. 
        """
        # unpack prediciton information
        mu, Sigma, pred_time = prediction

        # time until we get to the when the predictions are relevant
        time_delta = (pred_time - my_time).to_sec()

        # Calculate positions that are occupied
        ellipse = self.disallowed_positions(mu, Sigma)

        # calculate the projected position from following this velocity
        x = my_x + velocity[0]*time_delta
        y = my_y + velocity[1]*time_delta

        # Now we normalize the ellipse
        xc = x - ellipse.center[0]
        yc = y - ellipse.center[1]
        cos_angle = np.cos(np.radians(180.-ellipse.angle))
        sin_angle = np.sin(np.radians(180.-ellipse.angle))
        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle

        # and check distance to the unit circle
        r = (xct**2/(ellipse.width/2.)**2) + (yct**2/(ellipse.height/2.)**2)

        return r

    def disallowed_positions(self, mu, Sigma):
        """
        Find positions that will lead to a collision with
        probability over a given threshold.
        """
        # If we stay this Mahalanobis distance from the predicted obstacle
        # position, we should be safe
        safe_m_distance = np.sqrt(-2*np.log(1-self.theta))

        # Now convert this distance to an ellipse
        (W,v) = np.linalg.eig(Sigma)
        lambda1 = W[0]
        lambda2 = W[1]
        v1 = v[:,0]
        v2 = v[:,1]
        angle = np.angle(v1[0]+v1[1]*1j, deg=True)  # get the angle from the eigenvalues
        width = safe_m_distance*np.sqrt(lambda1) + self.safe_radius
        height = safe_m_distance*np.sqrt(lambda2) + self.safe_radius
        center = (mu[0], mu[1])
        ellipse = patches.Ellipse(center, width, height, angle=angle)

        # Points in this ellipse will collide: points outside it are safe
        return ellipse

    def in_ellipse(self, x, y, ellipse):
        """
        Check if a point (x,y) is contained in an ellipse
        """
        # normalize the ellipse
        xc = x - ellipse.center[0]
        yc = y - ellipse.center[1]
        cos_angle = np.cos(np.radians(180.-ellipse.angle))
        sin_angle = np.sin(np.radians(180.-ellipse.angle))
        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle

        # and check distance to the unit circle
        r = (xct**2/(ellipse.width/2.)**2) + (yct**2/(ellipse.height/2.)**2)

        return (r<=1)


if __name__=="__main__":
    robot_name = sys.argv[1]
    obstacle_name = sys.argv[2]

    # Wait a few seconds to be sure the predictor is up and running
    rospy.sleep(5)

    controller = DynamicCAController(robot_name, obstacle_name, steps=10, theta=0.001)
    controller.NPVO_verification()
    
