# RL-SLAM
A new method for robots to SLAM based on reinforce learning algorithms

Version of development environment: Ubuntu16.04, Cuda11.0, ROS-kinetic(python2.7), conda4.10.3(python3.7)

Place setup.sh inside your root folder, then run the following command:  
$ source ~/setup.sh

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

Since we have to use models in others' project, please donwload following packages and place them in your workspace:  
$ cd ~/catkin_ws/src  
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git  
$ git clone https://github.com/aws-robotics/aws-robomaker-small-house-world.git  
$ cd ~/catkin_ws  
$ catkin_make

Install OpenAI ROS package:  
$ cd ~/catkin_ws/src  
$ git clone https://github.com/edowson/openai_ros.git  
$ cd ~/catkin_ws  
$ catkin_make
$ source devel/setup.bash
$ rosdep install openai_ros  
After installing OpenAI ROS package, we have to add our own taskenv to the package.  
Place turtlebot2_house.py inside the /openai_ros/openai_ros/src/openai_ros/task_envs/turtlebot2 folder.

If there are any other packages not installed in your development environment, such as gym and torch, please install them by pip install.  
Also, do not forget to install them in the python3 virtual environment.

Start training:  
Place robot_control.sh and training.sh inside your root folder, then run the following command:  
$ source ~/robot_control.sh  
$ source ~/training.sh

Currently, we only have limited number of training scenes, to add more scenes for training, please follow these steps:  
1. Place your world file inside the /rrt_exploration_tutorials/launch/includes/worlds folder  
2. Add a launch file for your world inside the /rrt_exploration_tutorials/launch folder, the content of your launch file should be as same as other launch files under this folder except in line 7 you have to change the value of the arg "world_name" to the name of your world file  
3. Then edit the training.sh inside your root folder. I use world_array to store the remain training times of each scene and rand() to generate random numbers as the index of operations to launch a world within the loop. Please change them accordingly.

Other tips:  
1. If you wish to change your SLAM algorithms, please edit move_baseSafe.launch inside the /rrt_exploration_tutorials/launch/includes folder  
2. If you wish to change your Navigation algorithms, please edit base_global_planner_params.yaml and base_local_planner_params.yaml inside the /rrt_exploration_tutorials/param folder  
3. If you wish to change some params of your robot, please edit kobuki.urdf.xacro and kobuki_gazebo.urdf.xacro inside the /rrt_exploration_tutorials/launch/includes/urdf folder  
