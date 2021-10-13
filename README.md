# RL-SLAM
A new method for robots to SLAM based on reinforce learning algorithms

Version of development environment: Ubuntu16.04, Cuda11.0, ROS-kinetic(python2.7), conda4.10.3(python3.7)

Add following contents in  ~/.bashrc:   
$ export LD_LABRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64    
$ export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.0  
$ export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH  
$ export ROS_WORKSPACE=/opt/ros/kinetic/share:/opt/ros/kinetic/share/ros:/home/chenxr/catkin_ws/devel/share  
$ export PATH=/usr/local/cuda-11.0/bin:/opt/ros/kinetic/bin:~/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin   
$ export PYTHONPATH=/usr/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages  
$ export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:~/catkin_ws/  
$ export TURTLEBOT3_MODEL=waffle
$ source ~/.bashrc  

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


