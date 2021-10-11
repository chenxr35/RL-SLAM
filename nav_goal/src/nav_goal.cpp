#include "ros/ros.h"
#include "std_msgs/String.h"
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <iostream>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <move_base_msgs/MoveBaseActionGoal.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/MapMetaData.h>
#include <kobuki_msgs/BumperEvent.h>
#include <nav_msgs/Odometry.h>
#include <vector>
#include <string>

#define H 481
#define W 481
#define SCALE 0.1
#define LEFT  0
#define CENTER  1
#define RIGHT  2
#define INIT 996
#define SHUTDOWN 997
#define SEND 998
#define RECEIVE 999
#define WAIT 1000
#define GO 1001
#define BUMP 1002
#define TURNLEFT 1003
#define TURNRIGHT 1004

using namespace std;

#define FILENAME "/home/chenxr/Generating_global_goals/goal.txt"
#define MAPFILE "/home/chenxr/Generating_global_goals/global_map.txt"

class SubscribeAndPublish
{
public:
  SubscribeAndPublish(): map_signal(false), map_signal_(false), map_signal_pose(false), bump_signal(false), path_signal(true), search_signal(false), pose_x(0.0), pose_y(0.0), curr_x(1000.0), curr_y(1000.0), bump_times(0)
  {
    //Topic you want to publish
    pub_ = n_.advertise<move_base_msgs::MoveBaseActionGoal>("/robot_1/move_base/goal", 1000);
 
    //Topic you want to subscribe
    //sub_ = n_.subscribe("/robot_1/cmd_vel", 1, &SubscribeAndPublish::callback, this);

    goal_sub_ = n_.subscribe("/robot_1/move_base/goal", 1, &SubscribeAndPublish::Goalcallback, this);

    path_sub_ = n_.subscribe("robot_1/move_base_node/NavfnROS/plan", 1000, &SubscribeAndPublish::Pathcallback, this);
    
    map_sub_ = n_.subscribe("/robot_1/map", 1000000, &SubscribeAndPublish::mapCallback, this);
    
    map_pose_sub_ = n_.subscribe("/robot_1/map_metadata", 1, &SubscribeAndPublish::mapposeCallback, this);

    bumper_sub_ = n_.subscribe("/robot_1/mobile_base/events/bumper", 1, &SubscribeAndPublish::bumperCallBack, this);

    pose_sub_ = n_.subscribe("/robot_1/odom", 1, &SubscribeAndPublish::poseCallback, this);
  }

  void updatecommandGoal(float x, float y, int seq)
  {
    move_base_msgs::MoveBaseActionGoal aim;
    aim.goal.target_pose.header.frame_id = "robot_1/map";
    aim.goal.target_pose.header.seq = seq;
    aim.goal.target_pose.pose.position.x = x;
    aim.goal.target_pose.pose.position.y = y;
    aim.goal.target_pose.pose.position.z = 0.0;
    aim.goal.target_pose.pose.orientation.x = 0.0;
    aim.goal.target_pose.pose.orientation.y = 0.0;
    aim.goal.target_pose.pose.orientation.z = 0.0;
    aim.goal.target_pose.pose.orientation.w = 1.0;
    pub_.publish(aim);
  }

  void EndPath()
  {
    // end_signal = true;
    // ROS_INFO("End of the Path");
    map_signal_ = false;
    ROS_INFO("Allowed to load new map");
    map_signal_pose = false;
    ROS_INFO("Allowed to read new map pose");
    map_signal = false;
    ROS_INFO("Allowed to get new global map");
    updatecommandGoal(curr_x, curr_y, SEND);
    curr_x = 1000.0;
    curr_y = 1000.0;
    ROS_INFO("Reset feasible goal");
    bump_times = 0;
    ROS_INFO("Reset bumping times");
    // goal_signal = false;
    // ROS_INFO("Stop feasible goal fetching");
    search_signal = false;
    ROS_INFO("Stop searching free space");
    bump_signal = true;
    ROS_INFO("Stop detecting bump");
  }

  void Goalcallback(const move_base_msgs::MoveBaseActionGoal::ConstPtr& msg)
  {
    //std::cout<<"goal_signal:"<<goal_signal<<" end_signal:"<<end_signal<<std::endl;
    // if(msg->goal.target_pose.header.seq == RECEIVE && goal_signal == false && end_signal == true)
    if(msg->goal.target_pose.header.seq == RECEIVE && search_signal == false)
    {
      ROS_INFO("Receiving goal");
      float x = msg->goal.target_pose.pose.position.x;
      float y = msg->goal.target_pose.pose.position.y;
      updatecommandGoal(x, y, WAIT);// selected goal
      ROS_INFO("Selected goal published");
      // end_signal = false;
      // ROS_INFO("Start a new path");
      path_signal = true;
      ROS_INFO("Possible path exists");
      temp_x = x;
      temp_y = y;
      std::cout<<"Currently selected goal:"<<temp_x<<" "<<temp_y<<std::endl;
      search_signal = true;
      ROS_INFO("Allowed to search the goal in free space");
      // goal_signal = true;
      // ROS_INFO("Allowed to find the feasible goal");
    }
  }

  void poseCallback(const nav_msgs::Odometry::ConstPtr& msg)
  {
    // ROS_INFO("Receiving pose");
    // if(bump_signal != true && goal_signal == false)
    if(bump_signal != true)
    {
      pose_x = msg->pose.pose.position.x;
      pose_y = msg->pose.pose.position.y;
      if(curr_x != 1000.0 && curr_y != 1000.0){
        if(fabs(int(pose_x - curr_x)) > 1 || fabs(int(pose_y - curr_y)) > 1){
          ROS_INFO("Go on");
          updatecommandGoal(curr_x, curr_y, GO);// republish feasible goal
          sleep(3);
        }
        else{
          EndPath();
          ROS_INFO("Arrived");
        }
      }
    }
  }

  void bumperCallBack(const kobuki_msgs::BumperEvent::ConstPtr& msg)
  { 
    ROS_INFO("Receiving bumper");
    if(bump_signal == false)
    {
      if(msg->bumper == LEFT)
      {
        ROS_INFO("Bumping on the left");
        bump_times++;
        //ROS_INFO("Turn right");
        //updatecommandGoal(curr_x, curr_y, TURNRIGHT);
        //sleep(3);
        ROS_INFO("Go on");
        updatecommandGoal(curr_x, curr_y, GO);// republish feasible goal
        sleep(10);
      }
      else if(msg->bumper == CENTER)
      {
        ROS_INFO("Bumping on the center");
        bump_times++;
        //ROS_INFO("Turn right");
        //updatecommandGoal(curr_x, curr_y, TURNRIGHT);
        //sleep(3)
        ROS_INFO("Go on");
        updatecommandGoal(curr_x, curr_y, GO);// republish feasible goal
        sleep(10);
      }
      else if(msg->bumper == RIGHT)
      {
        ROS_INFO("Bumping on the right");
        bump_times++;
        //ROS_INFO("Turn left");
        //updatecommandGoal(curr_x, curr_y, TURNLEFT);
        //sleep(3);
        ROS_INFO("Go on");
        updatecommandGoal(curr_x, curr_y, GO);// republish feasible goal
        sleep(10);
      }
      if(bump_times >= 5)
      {
        bump_signal = true;
        ROS_INFO("Bumping confirmed");
        updatecommandGoal(curr_x, curr_y, BUMP);
        ROS_INFO("Please hold on");
        EndPath();
        ROS_INFO("Not a good path, stop moving");
      }
    }
  }
  /*
  void callback(const geometry_msgs::Twist::ConstPtr& msg)
  {
    if(msg->linear.x == 0 && msg->angular.z == 0 && goal_signal == false && end_signal == true)
    {
      ROS_INFO("Waiting for the goal");
      while(access(FILENAME, F_OK) == -1)
        continue;
      ROS_INFO("Goal file accessible");
      float *ptr;
      ptr = new float[2];

      while(1){
        fp.open(FILENAME, std::ios::in);
        for(int i = 0;i < 2;i++)
          fp>>ptr[i];
        fp.close();
        if(ptr[0] <= 1.e-30 && ptr[0] >= -1.e-30){
          ROS_INFO("Fail to load");
          continue;
        }
        else{
          ROS_INFO("Succeed to load");
          break;
        }
      }
      
      ROS_INFO("Loading the goal");
      std::cout<<"x:"<<ptr[0]<<std::endl;
      std::cout<<"y:"<<ptr[1]<<std::endl;
      remove(FILENAME);
      ROS_INFO("Goal file removed");
      updatecommandGoal(ptr[0], ptr[1], WAIT);// selected goal
      ROS_INFO("Selected goal published");
      end_signal = false;
      ROS_INFO("Start a new path");
      path_signal = true;
      ROS_INFO("Possible path exists");
      temp_x = ptr[0];
      temp_y = ptr[1];
      std::cout<<"Currently selected goal:"<<temp_x<<" "<<temp_y<<std::endl;
      goal_signal = true;
      ROS_INFO("Allowed to find the feasible goal");
      delete []ptr;
    }
    linear = msg->linear.x;
    angular = msg->angular.z;
  }
  */
  void Pathcallback(const nav_msgs::Path::ConstPtr& msg)
  {
    ROS_INFO("Receiving path");
    // std::cout<<"map_signal:"<<map_signal<<" goal_signal:"<<goal_signal<<" map_signal_:"<<map_signal_<<" map_signal_pose:"<<map_signal_pose<<" end_signal:"<<end_signal<<std::endl;
    if(msg->header.frame_id.size() == 0 && path_signal == true){
      updatecommandGoal(curr_x, curr_y, BUMP);
      ROS_INFO("Please hold on");
      path_signal = false;
      ROS_INFO("No possible path");
      EndPath();
      ROS_INFO("No way to go");
    }
    // if(map_signal == false && goal_signal == true && map_signal_ == true && map_signal_pose == true && end_signal == false && msg->header.frame_id.size() != 0)
    if(map_signal == false && map_signal_ == true && map_signal_pose == true && msg->header.frame_id.size() != 0 && search_signal == true)
    {
      ROS_INFO("Loading completed");
      //reshape the map
      vector<int>::iterator it;
      int map_data[map_H][map_W];
      for(int i = 0;i < map_H;i++)
        for(int j = 0;j < map_W;j++)
          map_data[i][j] = map_[j+(map_H-i-1)*map_W];//from left corner, row major
          //map_data[i][j] = map_[j*map_H+(map_H-i-1)];//from left corner, column major
          //map_data[i][j] = map_[(map_W-j-1)*map_H+(map_H-i-1)];//from right corner, column major
          //map_data[i][j] = map_[j+(map_H-i-1)*map_W];//from left corner, row major
      //get the global map 
      for(int i = 0;i < H;i++)
        for(int j = 0;j < W;j++)
          global_map[i][j] = 1;
      int map_origin_x, map_origin_y;
      //map_origin_x = world2map(origin_x, W);
      map_origin_x = int((H-1)/2 - origin_y/SCALE);
      if(map_origin_x > H - 1)
        map_origin_x =  H - 1;
      else if(map_origin_x < 0)
        map_origin_x = 0;
      //map_origin_y = world2map(origin_y, H);
      map_origin_y = int(origin_x/SCALE + (W-1)/2);
      if(map_origin_y > W - 1)
        map_origin_y =  W - 1;
      else if(map_origin_y < 0)
        map_origin_y = 0;
      std::cout<<"map_origin_x:"<<map_origin_x<<std::endl;
      std::cout<<"map_origin_y:"<<map_origin_y<<std::endl;
      int H_d, W_d;
      H_d = int(map_origin_x - map_H);
      W_d = int(map_origin_y);
      std::cout<<"H_d:"<<H_d<<" "<<"W_d:"<<W_d<<std::endl;
      for(int i = 0;i < map_H;i++)
        for(int j = 0;j < map_W;j++){
          if(i + H_d < H && j + W_d < W){
            if(map_data[i][j] == -1){
              global_map[i + H_d][j + W_d] = 1;
              //std::cout<<"unexplored area:"<<i<<" "<<j<<std::endl;
            }
            else if(map_data[i][j] == 0){
              global_map[i + H_d][j + W_d] = 0;
              std::cout<<"free area:"<<i+H_d<<" "<<j+W_d<<std::endl;
            }
            else if(map_data[i][j] == 100){
              global_map[i + H_d][j + W_d] = 1;
              std::cout<<"obstacle:"<<i+H_d<<" "<<j+W_d<<std::endl;
            }
          }
        }
      map_signal = true;
      ROS_INFO("Get the global map, not allowed to fetch new map currently");

      ROS_INFO("Selected goal has been published, allowed to fetch feasible goal");
      float x_ = 0;
      float y_ = 0;
      int id = 0;
      std::cout<<"x_:"<<x_<<" y_:"<<y_<<std::endl;
      std::cout<<"temp_x:"<<temp_x<<" temp_y:"<<temp_y<<std::endl;
      while(fabs(x_ - temp_x) > 0.3 || fabs(y_ - temp_y) > 0.3){
        std::cout<<"id:"<<id<<std::endl;
        x_ = msg->poses[id].pose.position.x;
        y_ = msg->poses[id].pose.position.y;
        std::cout<<"x_:"<<x_<<" "<<"y_:"<<y_<<std::endl;
        path_x.push_back(x_);
        path_y.push_back(y_);
        id++;
      }
      std::cout<<"id:"<<id<<std::endl;
      for(id = path_x.size() - 1;id >= 0;id--){
        std::cout<<"path_size:"<<id + 1<<std::endl;
        x_ = path_x[id];
        y_ = path_y[id];
        //convert world to map
        int x_map, y_map;
        x_map = int((H-1)/2 - y_/SCALE);
        if(x_map > H - 1)
          x_map =  H - 1;
        else if(x_map < 0)
          x_map = 0;
        y_map = int(x_/SCALE + (W-1)/2);
        if(y_map > W - 1)
          y_map = W - 1;
        else if(y_map < 0)
          y_map = 0;
        std::cout<<"x_map:"<<x_map<<" "<<"y_map:"<<y_map<<std::endl;
        std::cout<<"global_map:"<<global_map[x_map][y_map]<<std::endl;
        if(global_map[x_map][y_map] == 0){
          std::cout<<"id:"<<id<<std::endl;
        }
        else{
          std::cout<<"good id:"<<id<<std::endl;
          break;
        }
      }
      updatecommandGoal(path_x[id], path_y[id], GO);// feasible goal
      sleep(10);
      std::cout<<"Currently feasible goal:"<<path_x[id]<<" "<<path_y[id]<<std::endl;
      ROS_INFO("Feasible goal published");
      if(bump_signal == true)
      {
        ROS_INFO("Last path failed, allow to detect bumping again");
        bump_signal = false;
      }
      curr_x = path_x[id];
      curr_y = path_y[id];
      ROS_INFO("Feasible goal stored");
      search_signal = false;
      ROS_INFO("Stop searching free space");
      // goal_signal = false;
      // ROS_INFO("Stop feasible goal fetching");
    }
  }
  
  void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& map)
  {
    ROS_INFO("Receiving map");
    // std::cout<<"map_signal:"<<map_signal<<" goal_signal:"<<goal_signal<<" end_signal:"<<end_signal<<" map_signal_:"<<map_signal_<<std::endl;
    // if(map_signal == false && goal_signal == true && end_signal == false && map_signal_ == false)
    if(map_signal == false && map_signal_ == false)
    {
      int length = map->info.height * map->info.width;
      int data[length];
      for(unsigned int y = 0;y < map->info.height;y++){
        for(unsigned int x = 0;x < map->info.width;x++){
          unsigned int i = x + (map->info.height - y - 1)*map->info.width;
          data[i] = map->data[i];
        }
      }
      std::cout<<"Map height:"<<map->info.height<<std::endl;
      std::cout<<"Map width:"<<map->info.width<<std::endl;
      for(int i = 0;i < length;i++)
        map_.push_back(data[i]);
      map_signal_ = true;
      ROS_INFO("Map loaded");
    }
  }

  void mapposeCallback(const nav_msgs::MapMetaData::ConstPtr& msg){
    ROS_INFO("Receiving mappose");
    // std::cout<<"map_signal:"<<map_signal<<" goal_signal:"<<goal_signal<<" end_signal:"<<end_signal<<" map_signal_pose:"<<map_signal_pose<<std::endl;
    // if(map_signal == false && goal_signal == true && end_signal == false && map_signal_pose == false)
    if(map_signal == false && map_signal_pose == false)
    {
      origin_x = msg->origin.position.x;
      origin_y = msg->origin.position.y;
      map_H = int(msg->height);
      map_W = int(msg->width);
      std::cout<<"height:"<<map_H<<endl;
      std::cout<<"width:"<<map_W<<endl;
      std::cout<<"origin_x:"<<origin_x<<endl;
      std::cout<<"origin_y:"<<origin_y<<endl;
      map_signal_pose = true;
      ROS_INFO("Map pose read");
    }
  }
  
  int world2map(float world, int M)
  {
    int M_d;
    M_d = (M - 1) / 2;
    int map;
    map = int(M_d + world/SCALE);
    if(map > M - 1)
      map = M - 1;
    else if(map < 0)
      map = 0;
    return map;
  }
  
private:
  ros::NodeHandle n_; 
  ros::Publisher pub_;
  //ros::Subscriber sub_;
  ros::Subscriber path_sub_;
  ros::Subscriber map_sub_;
  ros::Subscriber map_pose_sub_;
  ros::Subscriber bumper_sub_;
  ros::Subscriber pose_sub_;
  ros::Subscriber goal_sub_;
  int global_map[H][W];
  int map_H, map_W;// height and width of map
  int bump_times;// bumping times
  bool map_signal;// allow to accept map or not
  bool map_signal_;// map loaded or not
  bool map_signal_pose;// map pose loaded or not
  // bool goal_signal;// seek feasible goal or not
  // bool end_signal;// signal the end of the path
  bool bump_signal;// bump or not
  bool path_signal;// path possible or not
  bool search_signal;// search free space or not
  float temp_x, temp_y;// temporary goal
  float curr_x, curr_y;// currently feasible goal
  float pose_x, pose_y;// current position
  float origin_x, origin_y;// origin of map
  float linear, angular;// speed
  vector<int> map_;// map data
  vector<float> path_x;
  vector<float> path_y;
  ifstream fp;
 
};//End of class SubscribeAndPublish


int main(int argc, char **argv)
{
  ros::init(argc, argv, "nav_goal");
  SubscribeAndPublish Navigation;

  ros::spin();

  return 0;
}
