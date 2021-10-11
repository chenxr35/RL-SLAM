cd ~/catkin_ws
catkin_make

unset PYTHONPATH

conda activate py37

source ~/catkin_ws/devel/setup.sh

killall nav_goal
killall velocity_shift
killall my_turtlebot2_training

roslaunch rrt_exploration_tutorials robot_control.launch
