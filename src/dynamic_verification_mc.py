#!/usr/bin/env python

##
#
# An object that represents the Markov Chain we'll use
# to verify safety properties of a given controller
#
##

import igraph
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.spatial import distance
from control_real_time import predict_distribution
from network_variables import *

def probabilistic_threshold_controller(mu, sigma, xobs, xcurr, xdes):
    """
    Given a distribution of likely obstacle positions and a desired direction (xdes)
    move the robot in a way that minimizes deviation from the desired direction
    subject to a constraint on the probability of collision. 
    """

    obs_radius = 0.25
    my_radius = 0.25

    # Construct a random variable representing probable obstacle positions
    rv = multivariate_normal([xobs[0] + mu[0], xobs[1] + mu[1]], sigma[0:2, 0:2])

    # Calculate (approximate) probibility of collision using a square of cdfs
    rad = obs_radius + 2*my_radius
    top_right = lambda x : [x[0] + rad, xdes[1] + rad]
    bottom_right = lambda x : [x[0] + rad, xdes[1] - rad]
    top_left = lambda x : [x[0] - rad, xdes[1] + rad]
    bottom_left = lambda x :[x[0] - rad, xdes[1] - rad]

    coll_prob = lambda x : rv.cdf(top_right(x)) - rv.cdf(bottom_right(x)) - rv.cdf(top_left(x)) + rv.cdf(bottom_left(x))
    performance_cost = lambda x : (x[0] - xdes[0])**2 + (x[1]-xdes[1])**2

    # Calculate an optimal next position
    theta = 0.1   # maximum allowable probibility of collision
    cons = ({'type': 'ineq', 'fun': lambda x : theta - coll_prob(x)})
    res = minimize(performance_cost, xdes, constraints=cons, options={"maxiter": 900}, method="COBYLA")

    return res.x

def ignorant_controller(mu, sigma, xobs, xcurr, xdes):
    """
    Just move in the desired direction, regardless of anything else
    """
    next_position = [xcurr[0]+xdes[0], xcurr[1]+xdes[1]]
    return next_position

class DynamicVerificationMC:
    def __init__(self, num_layers, sess):  # must include a tensorflow session 

        self.num_branches = 4   # limit to prediction in four quadrants for now
        self.num_layers = num_layers
        self.num_nodes = (self.num_branches**(self.num_layers+1) -1 ) / (self.num_branches - 1)   # assuming a perfect tree

        initial_observations = np.asarray(
		[[[ 0.15 , -0.015,  1.   , -0.1  ]], 
		[[ 0.2  , -0.02 ,  1.   , -0.1  ]], 
		[[ 0.1  , -0.02 , -1.   , -0.1  ]], 
		[[-0.2  , -0.02 , -1.   , -0.1  ]], 
		[[-0.2  , -0.02 , -1.   , -0.1  ]], 
		[[-0.15 , -0.015, -1.   , -0.1  ]], 
		[[-0.2  , -0.02 , -1.   , -0.1  ]], 
		[[-0.2  , -0.02 , -1.   , -0.1  ]], 
		[[ 0.2  , -0.02 ,  1.   , -0.1  ]], 
		[[ 0.15 , -0.015,  1.   , -0.1  ]], 
		[[ 0.2  , -0.02 ,  1.   , -0.1  ]], 
		[[ 0.2  , -0.02 ,  1.   , -0.1  ]], 
		[[ 0.2  , -0.02 ,  1.   , -0.1  ]], 
		[[-0.1  , -0.02 , -1.   , -0.1  ]], 
		[[-0.2  , -0.02 , -1.   , -0.1  ]], 
		[[-0.2  , -0.02 , -1.   , -0.1  ]], 
		[[-0.2  , -0.02 , -1.   , -0.1  ]], 
		[[-0.2  , -0.02 , -1.   , -0.1  ]], 
		[[ 0.1  , -0.02 ,  1.   , -0.1  ]], 
		[[ 0.15 , -0.015,  1.   , -0.1  ]]])

        initial_observations = np.asarray([[[ 0, 0, 0, 0 ]]])


        obstacle_initial_position = [0, 0]
        robot_initial_position = [0,1]
        
        self.node_list = []

	# create an tree graph with all the nodes and edges we're after
        g = igraph.Graph.Tree(self.num_nodes, self.num_branches)

        # create the root node
        root = DynamicVerificationNode(0, None, None)
        root.observations = initial_observations
        root.robot_position = robot_initial_position   # initial robot position
        root.set_obstacle_position(obstacle_initial_position)
        
        robot_desired_direction = [0, -0.1]

        self.node_list.append(root)

        # create all other nodes
        num_parents = (self.num_branches**self.num_layers-1)/(self.num_branches-1)
        for parent_idx in range(num_parents):
            print("==> Parent %s of %s" % (parent_idx,num_parents))
            
            parent = self.get_node(parent_idx)
            children_idx = [i[1] for i in g.get_edgelist() if i[0] == parent_idx]
           
            # make a first pass to describe the obstacle's motion
            for j in range(len(children_idx)):
                direction = j   # 0 = NE, 1 = NW, 2 = SW, 3 = SE
                
                c = DynamicVerificationNode(children_idx[j], parent, direction)
                c.set_observations()
                c.set_obstacle_position(obstacle_initial_position)
              
                parent.children.append(c)
                self.node_list.append(c)
            
            parent.set_transition_probabilities(sess)
           
            # make a second pass to define the robot's motion
            for j in range(len(children_idx)):
                c.set_robot_position(probabilistic_threshold_controller, robot_desired_direction)   
                #c.set_robot_position(ignorant_controller, robot_desired_direction)   
                c.set_label()


    def __str__(self):
        s = "***** DynamicVerificationMC *****\n"
        for node in self.node_list:
            if node.children:   # don't print empty transitions
                s += "    %s ---> %s \n" % (node.index, [c.index for c in node.children])
        s += "*********************************\n"
        return s

    
    def get_node(self, idx):
        """
        Returns the node with a given index
        """
        for node in self.node_list:
            if node.index == idx:
                return node
        print("Warning: No such node %s found!" % idx)
        return None

    def save_prism_model(self, filename):
        print("Saving model to %s " % filename)

        mc_string =  "dtmc\n\nmodule dynamic_obstacle_model\n\n"  # string that describes all nodes and edges in the graph

        mc_string += "\n    s : [0..%d] init 0;\n" % (self.num_nodes-1)   # states
        mc_string += "    c : [0..1] init 0;\n\n"                    # labels

        # Traverse the whole model, getting information about each transition
        for parent_idx in range( (self.num_branches**self.num_layers-1)/(self.num_branches-1) ):

            parent = self.get_node(parent_idx)

            mc_string += "    [] s=%d -> " % parent.index
            for i in range(len(parent.children)-1):
                child = parent.children[i]
                # update the line for the first children
                mc_string += "%f : (s'=%d) & (c'=%d) + " % (parent.transition_probs[i], child.index, child.label)
            # update for the last child, which ends the line
            mc_string += "%f : (s'=%d) & (c'=%d);\n" % (parent.transition_probs[-1], parent.children[-1].index, parent.children[-1].label)

        mc_string += "\n\nendmodule\n"

        with open(filename, "wb") as out:
            out.write(mc_string)


class DynamicVerificationNode:
    def __init__(self, idx, parent, direction):

        self.index = idx    
        self.parent = parent

        # parameters to describe probabale next position (position + velocity)
        self.mu = None
        self.sigma = None

        self.children = []
        self.transition_probs = []    # transition probabilities for each child node

        self.obstacle_position = None
        self.robot_position = None
        self.label = 0                # 0 --> no collision, 1 --> collision

        self.observations = None    
        self.direction = direction   # an integer that indicating which grid cell we just transitioned to
                                     # for now, 0=NE, 1=NW, 2=SW, 3=SE

    def __str__(self):
        s = "\n***** DynamicVerificaitonNode ****\n"
        s += "    Index: %s\n" % self.index
        s += "    Label: %s\n" % self.label
        s += "    Children: %s\n"  % [i.index for i in self.children]
        s += "    Transitions: %s\n"  % [i for i in self.transition_probs]
        s += "    Obstacle Position: %s\n" % self.obstacle_position
        s += "    Robot Position: %s\n" % self.robot_position
        s += "    Collision: %s\n" % self.label
        s += "**********************************\n"

        return s

    def set_observations(self):
        if self.direction == 0:
            # North-east
            new_observation = [[ 0.2, 0.02, 1, 0.1 ]]   # CAUTION: this makes some assumptions on 
                                                        # what it means to transition to each quadrant.
                                                        # These should be removed when we (eventually) use 
                                                        # a more systematic gridding approach
        elif self.direction == 1:
            # North-west
            new_observation = [[ -0.2, 0.02, -1, 0.1 ]]
        elif self.direction == 2:
            new_observation = [[ -0.2, -0.02, -1, -0.1 ]]
            # South-west
        elif self.direction == 3:
            new_observation = [[ 0.2, -0.02, 1, -0.1 ]]
            # South-east
        else:
            print("Warning: invalid direction %s" % self.direction)
            return 1

        # update observations without replacement
        self.observations = np.append(self.parent.observations, [new_observation], axis=0)

    def set_transition_probabilities(self, sess, num_samples=10):
        """
        Given a trained tensorflow model (sess), set transition probabilities for each
        child node.

        TODO: fix problem with occasional NAN values
        """
        assert( len(self.children) == 4 )
        assert( self.observations.size )   # observations must be non-empty

        # Get a predicted distribution over positions and velocities
        self.mu, self.sigma = predict_distribution(self.observations, num_samples, sess)
        # Construct a random variable that represents predicted change in position
        rv = multivariate_normal(self.mu[0:2], self.sigma[0:2,0:2])

        # get transition probabilities for four quadrants (q1 --> NE, q2 --> NW, q3 --> SW, q4 --> SE)
        q3 = rv.cdf([0,0])
        q2 = rv.cdf([0,1e8]) - q3
        q4 = rv.cdf([1e8,0]) - q3
        q1 = rv.cdf([1e8,1e8]) - (q2+q3+q4)
        tp = [q1, q2, q3, q4]

        self.transition_probs = [ tp[i.direction] for i in self.children ]

    def set_obstacle_position(self, initial_position):
        """
        Update the location of the obstacle in 2D cartesian space, 
        which we can use to check collision status.
        """
        assert(self.observations is not None)
        
        x = initial_position[0] + sum(self.observations[:,0,0])   # add all position deltas to the initial position
        y = initial_position[1] + sum(self.observations[:,0,1])

        self.obstacle_position = [x, y]

    def set_robot_position(self, control_fcn, desired_direction):
        """
        Update the position of the robot in 2D cartesian space. 
        control_fcn must be a function that returns the next step of the robot
        as a (deltax, deltay) tuple. 

        mu and sigma should describe the obstacles future predicted position
        based on the *last* set of observations (not the current state). 
        """

        if self.parent.robot_position is None:
            # Hack to ensure that the robot position of the parent is always defined
            self.parent.set_robot_position(control_fcn, desired_direction)

        self.robot_position = control_fcn(
                self.parent.mu, 
                self.parent.sigma, 
                self.parent.obstacle_position,
                self.parent.robot_position, 
                desired_direction)

    def set_label(self):
        """
        Set whether or not a certain state contains a collision
        """
        assert( self.robot_position is not None and self.obstacle_position is not None)

        obs_radius = 0.25
        robot_radius = 0.25
        collision_radius = obs_radius + 2*robot_radius

        dist = distance.euclidean(self.robot_position, self.obstacle_position)

        if dist < collision_radius:
            self.label = 1
        else:
            self.label = 0


if __name__=="__main__":

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "/home/vjkurtz/catkin_ws/src/rnn_collvoid/tmp/LSTM_saved_model")

        a = DynamicVerificationMC(6,sess)
        a.save_prism_model("test.pm")
