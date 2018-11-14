# Predicting Obstacle Motion with a Recurrent Network for Dynamic Collision Avoidance

![Multi-agent Collision Avoidance Image](multiagend_demo.png)

An LSTM network predicts future obstacle positions based on realtime predictions and uses these predictions for collision avoidace.
Further details are available in [this paper](https://arxiv.org/abs/1811.01075).

### Dependencies

- python
- ROS Kinetic
- Stage simulator
- stage\_ros
- tensorflow
- GPy

### Installation

Install the dependencies

Clone this repo to your catkin workspace source folder
`git clone https://github.com/vincekurtz/rnn_collvoid`

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

