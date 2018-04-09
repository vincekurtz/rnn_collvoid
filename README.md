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

Train the network (and perform some brief tests) with `src/train_lstm.py`

Perform more extensive tests on a trained model with `src/test_trained_model.py`

Test out a controller based on a trained model with `test_controller.launch`



