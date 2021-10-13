# RL-SLAM
A new method for robots to SLAM based on reinforce learning algorithms

Version of development environment: Ubuntu16.04, Cuda11.0, ROS-kinetic(python2.7), conda4.10.3(python3.7)

Place setup.sh inside your root folder, then run the following command:
source ~/.setup.sh

Create a workspace:  
$ mkdir -p ~/catkin_ws/src  
$ cd ~/catkin_ws/src  
$ catkin_init_workspace  
$ cd ~/catkin_ws  
$ catkin_make     
$ echo $ROS_PACKAGE_PATH  

Download this package:  
$ git clone https://github.com/chenxr35/RL-SLAM.git

Find following packages inside the downloaded package, place them inside the /src folder in your workspace:  
init_shutdown_world, my_turtlebot2_training, rrt_exploration_tutorials, velocity_shift  
And then compile using catkin_make.  

Create a python3 virtual environment named py37:  
$ conda create -n py37 python=3.7


