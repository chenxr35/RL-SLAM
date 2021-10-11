function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))
}

cd ~/catkin_ws
catkin_make
source ~/catkin_ws/devel/setup.sh

world_array=(10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10)

for i in {1..100}
do
    rnd=$(rand 1 16)
    while ((${world_array[$(($rnd-1))]}<=0))
    do
        rnd=$(rand 1 16)
    done
    killall rviz
    killall gzserver && killall gzclient
    if (($rnd==1));then
        world_array[0]=$((${world_array[0]}-1))
        roslaunch rrt_exploration_tutorials single_simulated_house.launch # valid world
    elif (($rnd==2));then
        world_array[1]=$((${world_array[1]}-1))
        # roslaunch rrt_exploration_tutorials single_simulated_largeMap.launch # invalid world
    elif (($rnd==3));then
        world_array[2]=$((${world_array[2]}-1))
        # roslaunch rrt_exploration_tutorials single_simulated_MTR.launch # possibly invalid world, too small
    elif (($rnd==4));then
        world_array[3]=$((${world_array[3]}-1))
        # roslaunch rrt_exploration_tutorials single_simulated_bharath.launch # too small
    elif (($rnd==5));then
        world_array[4]=$((${world_array[4]}-1))
        # roslaunch rrt_exploration_tutorials single_simulated_corridor.launch # too small
    elif (($rnd==6));then
        world_array[5]=$((${world_array[5]}-1))
        roslaunch rrt_exploration_tutorials single_simulated_house1.launch # valid world
    elif (($rnd==7));then
        world_array[6]=$((${world_array[6]}-1))
        # roslaunch rrt_exploration_tutorials single_simulated_myhouse_ball.launch # robot not correctly located
    elif (($rnd==8));then
        world_array[7]=$((${world_array[7]}-1))
        # roslaunch rrt_exploration_tutorials single_simulated_myhouse.launch # robot not correctly located
    elif (($rnd==9));then
        world_array[8]=$((${world_array[8]}-1))
        roslaunch rrt_exploration_tutorials single_simulated_my_office.launch # valid world
    elif (($rnd==10));then
        world_array[9]=$((${world_array[9]}-1))
        # roslaunch rrt_exploration_tutorials single_simulated_myOffice.launch # not a close space
    elif (($rnd==11));then
        world_array[10]=$((${world_array[10]}-1))
        roslaunch rrt_exploration_tutorials single_simulated_myworld_1.launch # valid world
    elif (($rnd==12));then
        world_array[11]=$((${world_array[11]}-1))
        # roslaunch rrt_exploration_tutorials single_simulated_myworld.launch # robot not correctly located, two small, not a close space
    elif (($rnd==13));then
        world_array[12]=$((${world_array[12]}-1))
        # roslaunch rrt_exploration_tutorials single_simulated_sample.launch # not a close space
    elif (($rnd==14));then
        world_array[13]=$((${world_array[13]}-1))
        roslaunch rrt_exploration_tutorials single_simulated_world.launch # valid world
    elif (($rnd==15));then
        world_array[14]=$((${world_array[14]}-1))
        export GAZEBO_MODEL_PATH=/home/chenxr/catkin_ws/src/aws-robomaker-small-house-world/models
        roslaunch rrt_exploration_tutorials single_simulated_small_house.launch # valid world
    elif (($rnd==16));then
        world_array[15]=$((${world_array[15]}-1))
        # export GAZEBO_MODEL_PATH=/home/chenxr/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models
        # roslaunch rrt_exploration_tutorials single_simulated_turtlebot3_house.launch # robot initially stucked
    fi
done

