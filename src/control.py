#!/usr/bin/env python

##
#
# Contains several methods for controlling a robot based on 
# predicted future positions
#
##

from online_predictor import *
from geometry_msgs.msg import Twist
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

class DynamicCAController():
    """
    A dynamic collision avoidance controller that 
    uses predictions of obstacle behavior to output
    safe velocities for another robot in the workspace.
    """
    def __init__(self, robot_name, prediction_object):
        # My location
        self.x = None
        self.y = None

        #rospy.init_node("controller")
        odom = rospy.Subscriber(robot_name + '/base_pose_ground_truth', Odometry, self.odom_callback)
        self.control_pub = rospy.Publisher(robot_name + '/cmd_vel', Twist, queue_size=10)

        # Get a dictionary of topics that hold predictions
        # a given number of steps into the future.
        prediction_subscribers = {}
        self.predictions = {}
        
        pred_pubs = prediction_object.prediction_publishers
        for i in pred_pubs:
            topic = pred_pubs[i].name
            prediction_subscribers[i] = rospy.Subscriber(topic, PoseWithCovarianceStamped, self.predict_callback, (i,))
            # predictions[i] will be set in the predictor callback
            self.predictions[i] = None  


        # Set the timestep and rate for prediction
        self.TIMESTEP = prediction_object.TIMESTEP
        self.rate = prediction_object.rate

    def predict_callback(self, data, args):
        """
        Store mean and covariance information from 
        the subscribed prediction topics.

        TODO: take header timestamps into account for greater precision
        """
        step_index = args[0]  # unpack from tuple
        mu = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        Sigma = np.array([[data.pose.covariance[0], data.pose.covariance[1]],
                          [data.pose.covariance[6], data.pose.covariance[7]]])

        self.predictions[step_index] = {"mu" : mu, "Sigma" : Sigma}

    def odom_callback(self, data):
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y

    def one_step_threshold_control(self, vdes):
        """
        Given the distribution of future positions at the next timestep,
        move the robot in a way that minimizes deviation from the desired
        velocity, subject to a constraint on the probability of collision.

        WARNING: Assumes that the robot is aligned in the x-direction of the world,
        and is fully acutated (can move in any direction at any time)
        """
        dt = self.TIMESTEP
        obstacle_radius = 0.125
        my_radius = 0.125
	theta = 0.1   # maximum allowable probibility of collision
    
        # Wait until we have our position and some predictions
        while (self.x is None) or (self.predictions[0] is None):
            self.rate.sleep()

        for i in range(100):

            # Calculate desired position
            xdes = [self.x + (vdes.linear.x*dt), self.y + (vdes.linear.y*dt)]
            
            # Construct a random variable representing probable obstacle position
            mu = self.predictions[0]["mu"]  # using only the first step
            Sigma = self.predictions[0]["Sigma"]
            rv = multivariate_normal(mu, Sigma)

            # Calculate an approximate (upper bound) probability of collision
            rad = obstacle_radius + 2*my_radius
            top_right = lambda x : [x[0] + rad, xdes[1] + rad]
            bottom_right = lambda x : [x[0] + rad, xdes[1] - rad]
            top_left = lambda x : [x[0] - rad, xdes[1] + rad]
            bottom_left = lambda x :[x[0] - rad, xdes[1] - rad]
            coll_prob = lambda x : rv.cdf(top_right(x)) - rv.cdf(bottom_right(x)) - rv.cdf(top_left(x)) + rv.cdf(bottom_left(x))

            # We want to minimize the distance between desired and actual velocities, thus this cost
            performance_cost = lambda x : (x[0] - xdes[0])**2 + (x[1]-xdes[1])**2

            # Calculate an optimal next position
            cons = ({'type': 'ineq', 'fun': lambda x : theta - coll_prob(x)})
            res = minimize(performance_cost, xdes, constraints=cons, options={"maxiter": 2000}, method="COBYLA")
            best_x = res.x

            if not res.success:
                print("Unable to find optimal solution!")
                print(res)

            # translate the best position into a command velocity
            cmd_vel = Twist()
            best_x_vel = (best_x[0] - self.x)/dt
            best_y_vel = (best_x[1] - self.y)/dt
            cmd_vel.linear.x = best_x_vel
            cmd_vel.linear.y = best_y_vel

            self.control_pub.publish(cmd_vel)

            self.rate.sleep()

        



if __name__=="__main__":
    try:
        # Start the predictor in one thread
        predictor_net = OnlinePredictionNetwork('robot_0', steps=2)
        p = threading.Thread(target=predictor_net.start_online_prediction)
        p.start()

        rospy.sleep(4)

        controller = DynamicCAController('robot_1', predictor_net)
        
        desired_velocity = Twist()
        desired_velocity.linear.y = -0.7

        controller.one_step_threshold_control(desired_velocity)
        
        # Keep rospy up so we can quit with ^C
        rospy.spin()

    except rospy.ROSInterruptException:
        # We pressed ^C: stop all threads in the predictor
        predictor_net.coord.request_stop()
    except:
        # Some other error occured: print the traceback too
        traceback.print_exc()
        predictor_net.coord.request_stop()
