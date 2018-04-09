#!/usr/bin/env python

##
#
# Record a dataset of robot motions to use
# for supervised training of the RNN.
#
# These may be a bit contrived, but for a real
# world (UTM) scenario we could collect actual
# data.
#
##

import rospy
import csv
from nav_msgs.msg import Odometry

# Store real-time data in a global variable to access asynchrnously
velocity_data = Odometry().twist.twist.linear   # linear x,y,z velocities
position_data = Odometry().pose.pose.position   # x,y,z position

def odom_callback(data):
    """
    Set position and velocity variables based on latest readings
    """
    global velocity_data
    global position_data

    velocity_data = data.twist.twist.linear
    position_data = data.pose.pose.position

if __name__=="__main__":

    filename = "/home/vjkurtz/catkin_ws/src/rnn_collvoid/data/test_data.csv"
    tao = 0.1   # time between samples, in seconds
    motion_data = []   # will store [[x,y,x',y'],...]

    try:
        rospy.init_node('rnn_data_collector', anonymous=False)
        odometer = rospy.Subscriber('/robot_0/base_pose_ground_truth', Odometry, odom_callback)

        while not rospy.is_shutdown():
            # Obtain position and velocity data, rounding to 4 decimals
            x = float("{0:.4f}".format(position_data.x))
            y = float("{0:.4f}".format(position_data.y))
            xdot = float("{0:.4f}".format(velocity_data.x))
            ydot = float("{0:.4f}".format(velocity_data.y))

            motion_data.append([x,y,xdot,ydot])

            # Wait the alotted duration
            rospy.sleep(tao)
        
    except rospy.ROSInterruptException:
        pass
    finally:
        print("saving to " + filename)
        # Save data to a CSV file
        with open(filename, 'w') as out:
            writer = csv.writer(out)

            # Write the header
            writer.writerow(["HDR  Collision Avoidance RNN Data"])
            writer.writerow(["HDR  tao = %s" % tao])
            writer.writerow(["HDR  x | y | xdot | ydot"])

            # And the actual data
            for row in motion_data:
                writer.writerow(row)
        print("done saving.")


