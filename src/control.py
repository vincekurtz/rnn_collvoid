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
import matplotlib.patches as patches

class NonlinearProbabilisticVelocityObstacle():
    """
    A representation of all velocities that will 
    lead to a collision in N timesteps, given the latest
    observations
    """
    def __init__(self, timestep, num_steps):
        self.dt = timestep
        self.N = num_steps

        # A list of ellipses representing positions
        # that will lead to a likely collision in the 
        # next N timesteps
        self.disallowed_positions = []

        # Current position, allows us to find safe velocities
        # while reasoning mostly in position space
        self.x = None
        self.y = None

    def find_safe_velocity(self, vdes):
        """
        Find the velocity that is closest to the
        preferred one, but outside of this velocity
        obstacle.
        """
        assert len(self.disallowed_positions) == self.N

        if self.is_safe(vdes):
            return vdes
        else:

            # Set the constraint that r >= 1 for safety.
            # Hardcoding all constraints for now
            cons = (
                {
                    'type' : 'ineq', 
                    'fun'  : lambda x : self.r_from_velocity(x[0],x[1],self.disallowed_positions[0],1) - 1,
                },
                {
                    'type' : 'ineq', 
                    'fun'  : lambda x : self.r_from_velocity(x[0],x[1],self.disallowed_positions[1],2) - 1,
                },
                {
                    'type' : 'ineq', 
                    'fun'  : lambda x : self.r_from_velocity(x[0],x[1],self.disallowed_positions[2],3) - 1,
                })

            # Set up the performance cost: minimize the difference between the chosen and 
            # desired velocities
            performance_cost = lambda x : (x[0] - vdes.linear.x)**2 + (x[1] - vdes.linear.y)**2

            res = minimize(performance_cost, [vdes.linear.x, vdes.linear.y], constraints=cons, options={"maxiter": 2000}, method="COBYLA")

            if not res.success:
                print(res)

            # Choose a safe desired velocity
            v_safe = Twist()
            v_safe.linear.x = res.x[0]
            v_safe.linear.y = res.x[1]

            #print(v_safe)
            print("")
            print(self.r_from_velocity(res.x[0],res.x[1], self.disallowed_positions[0], 1), cons[0]['fun'](res.x))
            print(self.r_from_velocity(res.x[0],res.x[1], self.disallowed_positions[1], 2), cons[1]['fun'](res.x))
            print(self.r_from_velocity(res.x[0],res.x[1], self.disallowed_positions[2], 3), cons[2]['fun'](res.x))
            #print(cons[1]['fun'](res.x))
            #print(cons[2]['fun'](res.x))


            return v_safe

    def max_r_from_velocity(self, vx, vy):
        """
        Returns the maximum normalized distance from the obstacle
        over all predicted timesteps. Getting this under 1 should
        ensure that a velocity is good for all predicted timesteps
        """
        max_r = max([self.r_from_velocity(vx, vy, self.disallowed_positions[i], (i+1)) for i in range(self.N)])
        return max_r

    def r_from_velocity(self, vx, vy, ellipse, timestep):
        """
        Given a velocity and an ellipse of unsafe positions,
        find a normalized distance, r, of the velocity from
        that ellipse in the given timestep.

        i.e. if r <= 1, the velocity is unsafe.
        """
        time_delta = self.dt*timestep  # the converstion between position and velocity depends
                                       # how far in the future we're looking
        
        xdes = [self.x + (vx*time_delta), self.y + (vy*time_delta)]
        
        x = xdes[0]
        y = xdes[1]

        ellipse = self.disallowed_positions[0]  # just using the first step for now

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

    def is_safe(self, velocity):
        """
        Indicate whether the given velocity is safe or not
        """
        for i in range(self.N):
            time_delta = (i+1)*self.dt
            xdes = [self.x + (velocity.linear.x*time_delta), self.y + (velocity.linear.y*time_delta)]
            if self.disallowed_positions[i].contains_point(xdes):
                return False
        return True




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
        self.timestep = prediction_object.TIMESTEP
        self.rate = prediction_object.rate

        # Define a safe distance from the obstacle
        obstacle_radius = 0.125
        my_radius = 0.125
        self.safe_radius = obstacle_radius + 2*my_radius

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
        dt = self.timestep
	theta = 0.1   # maximum allowable probibility of collision
    
        # Wait until we have our position and some predictions
        while (self.x is None) or (self.predictions[0] is None):
            self.rate.sleep()

        while not rospy.is_shutdown():  # keep applying the control indefinitely

            # Calculate desired position
            xdes = [self.x + (vdes.linear.x*dt), self.y + (vdes.linear.y*dt)]
            
            # Construct a random variable representing probable obstacle position
            mu = self.predictions[0]["mu"]  # using only the first step
            Sigma = self.predictions[0]["Sigma"]
            rv = multivariate_normal(mu, Sigma)

            # Calculate an approximate (upper bound) probability of collision
            coll_prob = lambda x : self.collision_probability(rv, x)

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

    def NPVO_control(self, vdes):
        """
        A controller based on the concept of a Nonlinear Probibalistic
        Velocity Obstacle. At each timestep, a velocity is chosen that
        is as close as possible to the desired velocity, but that
        will never lead to a collision (according to our current
        predictions, with a given probability). 
        """
        theta = 0.1 # maximum allowable probability of collision

        # Wait until we have our position and some predictions
        while (self.x is None) or (self.predictions[0] is None):
            self.rate.sleep()

        print("Beginning control sequence")
        while not rospy.is_shutdown():

            NPVO = NonlinearProbabilisticVelocityObstacle(self.timestep, len(self.predictions))

            for step in self.predictions:
                mu = self.predictions[step]["mu"]
                Sigma = self.predictions[step]["Sigma"]

                p_ellipse = self.find_disallowed_positions(mu, Sigma, theta)
                NPVO.disallowed_positions.append(p_ellipse)

            (NPVO.x, NPVO.y) = (self.x, self.y)
            best_vel = NPVO.find_safe_velocity(vdes)

            print(best_vel.linear)

            # Move the robot down
            self.control_pub.publish(best_vel)

            self.rate.sleep()

    def find_disallowed_positions(self, mu, Sigma, collision_threshold):
        """
        Find positions that will lead to a collision with
        probability over a given threshold.
        """
        # If we stay this Mahalanobis distance from the predicted obstacle
        # position, we should be safe
        safe_m_distance = np.sqrt(-2*np.log(1-collision_threshold))

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


if __name__=="__main__":
    try:
        # Start the predictor in one thread
        predictor_net = OnlinePredictionNetwork('robot_0', steps=3)
        p = threading.Thread(target=predictor_net.start_online_prediction)
        p.start()

        # Wait a few seconds to be sure the predictor is up and running
        rospy.sleep(2)

        controller = DynamicCAController('robot_1', predictor_net)
        
        desired_velocity = Twist()
        desired_velocity.linear.y = -0.7

        controller.NPVO_control(desired_velocity)
        
        # Keep rospy up so we can quit with ^C
        rospy.spin()

    except rospy.ROSInterruptException:
        # We pressed ^C: stop all threads in the predictor
        predictor_net.coord.request_stop()
    except:
        # Some other error occured: print the traceback too
        traceback.print_exc()
        predictor_net.coord.request_stop()
