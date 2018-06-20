# Time-Dependent Collision Avoidance based on a Recurrent Network

### Basic Idea

After observing obstacle positions and velocities for a while, an LSTM network predicts future obstacle positions and uses these predictions for collision avoidace

### Dependencies

- ROS Kinetic
- Stage simulator
- stage\_ros
- python
- tensorflow

### Usage

Make predictions in real-time, and make a plot of predictions afterwards:
```
roslaunch prediction_plot.launch
```

Make predictions in real-time, and use them to control a robot:
```
roslaunch predict_control.launch
```

With the simulation running, visualize what's happening with rviz:
```
roslaunch rnn_collvoid rviz.launch
```

