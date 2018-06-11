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

Launch a simulation with one oscillating obstacle and one controllable agent:
```
roslaunch rnn_collvoid oscillating_sim.launch
```

Control the robot with a NPVO:
```
rosrun rnn_collvoid control.py
```

Predict the motion of a moving obstacle, and view the predictions in rviz:
```
rosrun rnn_collvoid online_predictor.py
roslaunch rnn_collvoid rviz.launch
```

