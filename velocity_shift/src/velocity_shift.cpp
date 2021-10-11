#include "ros/ros.h"
#include "std_msgs/String.h"
#include <iostream>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <move_base_msgs/MoveBaseActionGoal.h>
#include <kobuki_msgs/BumperEvent.h>
#include "stdio.h"
#include <time.h>
#include <stdlib.h>
#define LEFT  0
#define CENTER  1
#define RIGHT  2
#define GET_TB2_DIRECTION 0
#define TB2_DRIVE_FORWARD 1
#define TB2_RIGHT_TURN    2
#define TB2_LEFT_TURN     3
#define LINEAR_VELOCITY  0.3
#define ANGULAR_VELOCITY 1.5
#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)
#define INIT 996
#define SHUTDOWN 997
#define SEND 998
#define RECEIVE 999
#define WAIT 1000
#define GO 1001
#define BUMP 1002
#define SPIN 1003
#define LEFT_RECOVERY 1004
#define CENTER_RECOVERY 1005
#define RIGHT_RECOVERY 1006

class SubscribeAndPublish
{
public:
  SubscribeAndPublish(): move_signal(false)
  {
    //Topic you want to publish
    pub_ = n_.advertise<geometry_msgs::Twist>("/robot_1/mobile_base/commands/velocity", 1000);
 
    //Topic you want to subscribe
    sub_ = n_.subscribe("/robot_1/cmd_vel", 1, &SubscribeAndPublish::callback, this);

    laser_scan_sub_  = n_.subscribe("/robot_1/base_scan", 10, &SubscribeAndPublish::laserScanMsgCallBack, this);

    odom_sub_ = n_.subscribe("/robot_1/odom", 10, &SubscribeAndPublish::odomMsgCallBack, this);

    goal_sub_ = n_.subscribe("/robot_1/move_base/goal", 1, &SubscribeAndPublish::goalCallBack, this);
   
    bumper_sub_ = n_.subscribe("/robot_1/mobile_base/events/bumper", 1, &SubscribeAndPublish::bumperCallBack, this);

    // initialize variables
    escape_range_       = 30.0 * DEG2RAD;
    check_forward_dist_ = 0.7;
    check_side_dist_    = 0.6;
  }

  void bumperCallBack(const kobuki_msgs::BumperEvent::ConstPtr& msg)
  {/* 
    if(msg->bumper == LEFT)
    {
      //updatecommandVelocity(0, ANGULAR_VELOCITY);
      ROS_INFO("Bumping on the left");
      updatecommandVelocity(-1 * LINEAR_VELOCITY, 0);
      updatecommandVelocity(0, 0);
      sleep(5);
      ROS_INFO("Step back and hold on");
    }
    else if(msg->bumper == CENTER)
    {
      ROS_INFO("Bumping on the center");
      updatecommandVelocity(-1 * LINEAR_VELOCITY, 0);
      updatecommandVelocity(0, 0);
      sleep(5);
      ROS_INFO("Step back and hold on");
    }
    else if(msg->bumper == RIGHT)
    {
      //updatecommandVelocity(0, -1 * ANGULAR_VELOCITY);
      ROS_INFO("Bumping on the right");
      updatecommandVelocity(-1 * LINEAR_VELOCITY, 0);
      updatecommandVelocity(0, 0);
      sleep(5);
      ROS_INFO("Step back and hold on");
    }*/
  }
 
  void callback(const geometry_msgs::Twist::ConstPtr& msg)
  {
    std::cout<<"move_signal:"<<move_signal<<std::endl;
    //int var[] = {-1,0,1};
    //srand((unsigned)time(NULL)); 
    //ROS_INFO("velocity_shifting");
    //geometry_msgs::Twist news;

    /*
    double shift_1 = rand()/(double)(RAND_MAX/100)/1000*var[rand()%(sizeof(var)/sizeof(*var))];
    //news.linear.x = msg->linear.x + shift_1;
    news.linear.x = msg->linear.x;
    
    double shift_2 = rand()/(double)(RAND_MAX/100)/1000*var[rand()%(sizeof(var)/sizeof(*var))];
    //news.angular.z = msg->angular.z + shift_2;
    news.angular.z = msg->angular.z;

    news.linear.y = msg->linear.y;
    news.linear.z = msg->linear.z;
    news.angular.x = msg->angular.x;
    news.angular.y = msg->angular.y;
    pub_.publish(news);
    */
    /*
    if(scan_data_[CENTER] > check_forward_dist_)
    {
      if(scan_data_[LEFT] < check_side_dist_)
      {
        prev_tb2_pose_ = tb2_pose_;
        if(fabs(prev_tb2_pose_ - tb2_pose_) < escape_range_){
          updatecommandVelocity(-0.1, 0.0);
          updatecommandVelocity(0.0, -1 * ANGULAR_VELOCITY);
          ROS_INFO("TURN RIGHT");
        }
      }
      else if (scan_data_[RIGHT] < check_side_dist_)
      {
        prev_tb2_pose_ = tb2_pose_;
        if(fabs(prev_tb2_pose_ - tb2_pose_) < escape_range_){
          updatecommandVelocity(-0.1, 0.0);
          updatecommandVelocity(0.0, -1 * ANGULAR_VELOCITY);
          ROS_INFO("TURN LEFT");
        }
      }
      else{
        updatecommandVelocity(msg->linear.x, msg->angular.z);
        ROS_INFO("MOVE ON");
      }
    }

    else if (scan_data_[CENTER] < check_forward_dist_)
    {
      prev_tb2_pose_ = tb2_pose_;
      if(fabs(prev_tb2_pose_ - tb2_pose_) < escape_range_){
        updatecommandVelocity(-0.1, 0.0);
        updatecommandVelocity(0.0, -1 * ANGULAR_VELOCITY);
        ROS_INFO("TURN RIGHT");
      }
    }
    */
    if(move_signal == true){
      updatecommandVelocity(msg->linear.x, msg->angular.z);
      ROS_INFO("MOVE ON");
    }
    else{
      updatecommandVelocity(0, 0);
      ROS_INFO("HOLD ON");
    }
  }

  void laserScanMsgCallBack(const sensor_msgs::LaserScan::ConstPtr &msg)
  {
    uint16_t scan_angle[3] = {0, 30, 330};
  
    for (int num = 0; num < 3; num++)
    {
      if (std::isinf(msg->ranges.at(scan_angle[num])))
      {
        scan_data_[num] = msg->range_max;
        //ROS_INFO("SCANNING:%f",scan_data_[num]);
      }
      else
      {
        scan_data_[num] = msg->ranges.at(scan_angle[num]);
        //ROS_INFO("SCANNING:%f",scan_data_[num]);
      }
    }
  }
  
  void odomMsgCallBack(const nav_msgs::Odometry::ConstPtr &msg)
  {
    double siny = 2.0 * (msg->pose.pose.orientation.w * msg->pose.pose.orientation.z + msg->pose.pose.orientation.x * msg->pose.pose.orientation.y);
    double cosy = 1.0 - 2.0 * (msg->pose.pose.orientation.y * msg->pose.pose.orientation.y + msg->pose.pose.orientation.z * msg->pose.pose.orientation.z);  

    tb2_pose_ = atan2(siny, cosy);
  }

  void updatecommandVelocity(double linear, double angular)
  {
    geometry_msgs::Twist cmd_vel;

    cmd_vel.linear.x  = linear;
    cmd_vel.angular.z = angular;

    pub_.publish(cmd_vel);
  }

  void goalCallBack(const move_base_msgs::MoveBaseActionGoal::ConstPtr &msg)
  {
    if(msg->goal.target_pose.header.seq == WAIT){
      move_signal = false;
      ROS_INFO("Waiting");
    }
    else if(msg->goal.target_pose.header.seq == BUMP){
      move_signal = false;
      ROS_INFO("Bumping");
    }
    else if(msg->goal.target_pose.header.seq == SEND){
      move_signal = false;
      ROS_INFO("Sending");
    }
    else if(msg->goal.target_pose.header.seq == RECEIVE){
      move_signal = false;
      ROS_INFO("Receiving");
    }
    else if(msg->goal.target_pose.header.seq == GO){
      move_signal = true;
      ROS_INFO("Going on");
    }
    else if(msg->goal.target_pose.header.seq == SHUTDOWN){
      move_signal = false;
      ROS_INFO("SHUTDOWN");
    }
    else if(msg->goal.target_pose.header.seq == INIT){
      move_signal = false;
      ROS_INFO("INIT");
    }
    else if(msg->goal.target_pose.header.seq == SPIN){
      move_signal = false;
      updatecommandVelocity(0, 1);
      ROS_INFO("RESET");
    }
    else if(msg->goal.target_pose.header.seq == LEFT_RECOVERY){
      move_signal = false;
      updatecommandVelocity(0, ANGULAR_VELOCITY);
      ROS_INFO("LEFT_RECOVERY");
    }
    else if(msg->goal.target_pose.header.seq == CENTER_RECOVERY){
      move_signal = false;
      updatecommandVelocity(-1 * LINEAR_VELOCITY, 0);
      ROS_INFO("CENTER_RECOVERY");
    }
    else if(msg->goal.target_pose.header.seq == RIGHT_RECOVERY){
      move_signal = false;
      updatecommandVelocity(0, -1 * ANGULAR_VELOCITY);
      ROS_INFO("RIGHT_RECOVERY");
    }
    else{
      move_signal = true;
      ROS_INFO("Going on");
    }
  }

private:
  ros::NodeHandle n_; 
  ros::Publisher pub_;
  ros::Subscriber sub_;
  ros::Subscriber laser_scan_sub_;
  ros::Subscriber odom_sub_;
  ros::Subscriber goal_sub_;
  ros::Subscriber bumper_sub_;
  double tb2_pose_;
  double prev_tb2_pose_;
  double scan_data_[3] = {1000.0, 1000.0, 1000.0};
  double escape_range_;
  double check_forward_dist_;
  double check_side_dist_;
  bool move_signal;
 
};//End of class SubscribeAndPublish

int main(int argc, char **argv)
{
  //Initiate ROS
  ros::init(argc, argv, "velocity_shift");
 
  //Create an object of class SubscribeAndPublish that will take care of everything
  SubscribeAndPublish SAPObject;
 
  ros::spin();

  return 0;
}
