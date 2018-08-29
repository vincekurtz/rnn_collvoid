# Time-Dependent Collision Avoidance based on a Recurrent Network

### Basic Idea

After observing obstacle positions and velocities for a while, an LSTM network predicts future obstacle positions and uses these predictions for collision avoidace

### Dependencies

- ROS Kinetic
- Stage simulator
- stage\_ros
- python
- tensorflow

### Installation

Install the dependencies

Clone this repo to your catkin workspace
`git clone https://github.com/vincekurtz/rnn_collvoid`

Initialize the do-mpc submodule
```
git submodule init
git submodule update
```

Build the project:
```
cd [catkin_ws]
catkin_make
```

### Usage

Make predictions in real-time, and make a plot of predictions afterwards:
```
roslaunch rnn_collvoid predict.launch
```

Make predictions in real-time, and use them to control a robot:
```
roslaunch rnn_collvoid control.launch
```

Use this prediction system on two robots to avoid a collision with each other
```
roslaunch rnn_collvoid multi_agent.launch
```

With the simulation running, visualize what's happening with rviz:
```
roslaunch rnn_collvoid rviz.launch
```

