# Action-Recognition-HSR

For more information about this project see the following paper: (in progress)

## Requirements:


  - Python 3.8.10
  - ROS Noetic (http://wiki.ros.org/noetic/Installation/Ubuntu)
  - OpenCV 
  - Numpy 
  - Mediapipe
  - Sklearn 
  - Keras 2.10.0
  - Tensorflow 2.10.0
    
## Steps to follow:

  1. Clone the repository in your catkin/src folder
  2. Execute "catkin_make" inside your catkin folder
  3. (Optional) Remove the existing actions in the "MP_data" folder to train the model from scratch.
  4. Set the "train" parameter to True to acivate the training mode and record the desired actions.
  5. Add/modify the PoseStamped message of each action inside the method "speak" to indicate where the robot has to go once the action it's been detected.
  6. Set the "train" parameter to False
  7. Initialize the Toyota HSR Gazeboo-based simulator (see instructions in https://github.com/hsr-project/tmc_wrs_docker), or deploy the real robot. 
  8. Execute the command "rosrun action_recognition_hsr action_recognizer.py" in a new terminal

**Note:** If you are using a real Toyota HSR robot instead of the simulator, you may need to scan the environment first in order to obtain a .pgm map file if you don't have it yet. 
