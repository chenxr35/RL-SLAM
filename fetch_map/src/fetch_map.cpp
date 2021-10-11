#include <cstdio>
#include "ros/ros.h"
#include <nav_msgs/GetMap.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/MapMetaData.h>
#include <geometry_msgs/Twist.h>
#include <move_base_msgs/MoveBaseActionGoal.h>
#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#define WAIT 1000
#define GO 1001
#define BUMP 1002

using namespace std;

string int2string(int &n)
{
  std::stringstream newstr;
  newstr<<n;
  return newstr.str();
}

string float2string(float &n)
{
  std::stringstream newstr;
  newstr<<n;
  return newstr.str();
}

class MapGenerator
{
  public:
    MapGenerator(const std::string& mapname) : mapname_(mapname), saved_map_(false), signal_(false), signal(false), sign(false), move_signal(false)
    {
      ros::NodeHandle n;
      ROS_INFO("Waiting for the map");
      map_sub_ = n.subscribe("/robot_1/map", 1000000, &MapGenerator::mapCallback, this);
      vel_sub_ = n.subscribe("/robot_1/cmd_vel", 1, &MapGenerator::velCallback, this);
      pose_sub_ = n.subscribe("/robot_1/odom", 1, &MapGenerator::poseCallback, this);
      map_pose_sub_ = n.subscribe("/robot_1/map_metadata", 1, &MapGenerator::mapposeCallback, this);
      goal_sub_ = n.subscribe("/robot_1/move_base/goal", 1, &MapGenerator::goalCallback, this);
    }

  void goalCallback(const move_base_msgs::MoveBaseActionGoal::ConstPtr &msg)
  {
    if(msg->goal.target_pose.header.seq == WAIT)
      move_signal = false;
    else if(msg->goal.target_pose.header.seq == GO)
      move_signal = true;
    else if(msg->goal.target_pose.header.seq == BUMP)
      move_signal = true;
    else
      move_signal = true;
  }
  
  void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& map)
  {
    if(signal_ == true){
      int length = map->info.height * map->info.width;
      int data[length];
      for(unsigned int y = 0;y < map->info.height;y++){
        for(unsigned int x = 0;x < map->info.width;x++){
          unsigned int i = x + (map->info.height - y - 1)*map->info.width;
          data[i] = map->data[i];
        }
      }
      string strs;
      string str;
      for(int i = 0;i < length;i++){
        int &temp = data[i];
          str = int2string(temp);
          strs += str;
          strs += " ";
      }
      int height = map->info.height;
      int &temp_1 = height;
      str = int2string(temp_1);
      strs += str;
      strs += " ";
      int width = map->info.width;
      int &temp_2 = width;
      str = int2string(temp_2);
      strs += str;
      ofstream fout;
      fout.open("/home/chenxr/Generating_global_goals/map.txt");
      ROS_INFO("New a map file");
      fout<<strs<<endl;
      fout.close();
      ROS_INFO("Done");
      signal_ = false;
      //saved_map_ = true;
    }
  }
  
  void poseCallback(const nav_msgs::Odometry::ConstPtr& msg){
    if(signal == true){
      string strs;
      string str;
      float x = msg->pose.pose.position.x;
      float y = msg->pose.pose.position.y;
      float z = msg->pose.pose.position.z;
      float &temp_1 = x;
      str = float2string(temp_1);
      strs += str;
      strs += " ";
      float &temp_2 = y;
      str = float2string(temp_2);
      strs += str;
      strs += " ";
      float &temp_3 = z;
      str = float2string(temp_3);
      strs += str;
      ofstream fout;
      fout.open("/home/chenxr/Generating_global_goals/pose.txt");
      ROS_INFO("New a pose file");
      fout<<strs<<endl;
      fout.close();
      signal = false;
    }
  }

  void mapposeCallback(const nav_msgs::MapMetaData::ConstPtr& msg){
    if(sign == true){
      string strs;
      string str;
      float x = msg->origin.position.x;
      float y = msg->origin.position.y;
      float z = msg->origin.position.z;
      float &temp_1 = x;
      str = float2string(temp_1);
      strs += str;
      strs += " ";
      float &temp_2 = y;
      str = float2string(temp_2);
      strs += str;
      strs += " ";
      float &temp_3 = z;
      str = float2string(temp_3);
      strs += str;
      ofstream fout;
      fout.open("/home/chenxr/Generating_global_goals/map_pose.txt");
      ROS_INFO("New a map_pose file");
      fout<<strs<<endl;
      fout.close();
      sign = false;
    }
  }

  void velCallback(const geometry_msgs::Twist::ConstPtr& msg)
  {
    if(msg->linear.x == 0 && msg->linear.y == 0 && msg->linear.z == 0 && msg->angular.x == 0 && msg->angular.y == 0 && msg->angular.z == 0 && move_signal == true)
    {
      signal_ = true;
      signal = true;
      sign = true;
    }
    else{
      signal_ = false;
      signal = false;
      sign = false;
    }
  }

  std::string mapname_;
  ros::Subscriber map_sub_;
  ros::Subscriber vel_sub_;
  ros::Subscriber pose_sub_;
  ros::Subscriber map_pose_sub_;
  ros::Subscriber goal_sub_;
  bool saved_map_;
  bool signal_;
  bool signal;
  bool sign;
  bool move_signal;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "fetch_map");
  std::string mapname = "map";
  
  MapGenerator mg(mapname);
  /*
  while(!mg.saved_map_ && ros::ok())
    ros::spinOnce();
  */
  while(ros::ok())
    ros::spinOnce();

  return 0;
}
