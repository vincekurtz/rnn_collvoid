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

Record data with `generate_data.launch`.

Train the network (and perform some brief tests) with `train_lstm.launch`



