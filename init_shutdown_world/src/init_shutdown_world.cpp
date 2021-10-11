#include "ros/ros.h"
#include <move_base_msgs/MoveBaseActionGoal.h>
#include <std_msgs/String.h>
#define INIT 996
#define SHUTDOWN 997

class SubscribeAndPublish
{
public:
    SubscribeAndPublish()
    {
        pub_ = n_.advertise<move_base_msgs::MoveBaseActionGoal>("/robot_1/move_base/goal", 1);
        sub_ = n_.subscribe("/robot_1/move_base/goal", 1, &SubscribeAndPublish::callback, this);
        end_sub_ = n_.subscribe("/end_cond", 1, &SubscribeAndPublish::endcallback, this);
        updatecommandGoal(0, 0, INIT);
    }

    void callback(const move_base_msgs::MoveBaseActionGoal::ConstPtr& msg)
    {
        if(msg->goal.target_pose.header.seq == SHUTDOWN)
        {
            ros::shutdown();
        }
    }

    void endcallback(const std_msgs::String::ConstPtr& msg)
    {
        ros::shutdown();
    }

    void updatecommandGoal(float x, float y, int seq)
    {
        move_base_msgs::MoveBaseActionGoal aim;
        aim.goal.target_pose.header.frame_id = "/robot_1/map";
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

private:
    ros::NodeHandle n_; 
    ros::Publisher pub_;
    ros::Subscriber sub_;
    ros::Subscriber end_sub_;
};


int main(int argc,char **argv) 
{
  ros::init(argc, argv, "init_shutdown_world");
  SubscribeAndPublish InitAndShutdown;
  ros::spin();

  return 0;
}
