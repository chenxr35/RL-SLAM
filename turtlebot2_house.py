#!/usr/bin/python

import rospy
import numpy
import time
import math
import torch
import gym
from gym import spaces
from openai_ros.robot_envs import turtlebot2_env
from gym.envs.registration import register
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseActionGoal
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from nav_msgs.msg import MapMetaData
from kobuki_msgs.msg import BumperEvent
from geometry_msgs.msg import PoseWithCovarianceStamped
import os

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
timestep_limit_per_episode = 100 # Can be any Value

register(
        id='TurtleBot2House-v0',
        entry_point='openai_ros.task_envs.turtlebot2.turtlebot2_maze:TurtleBot2HouseEnv',
        # timestep_limit=timestep_limit_per_episode,
    )

# devices config
device1 = (
    torch.device("cuda", 0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)
global_policy_gpu_ids = [0]
device = global_policy_gpu_ids[0]


# class TurtleBot2HouseEnv(turtlebot2_env.TurtleBot2Env):
class TurtleBot2HouseEnv(gym.Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot2 in some kind of house.
        It will learn how to move around the house without crashing.
        """

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot2HouseEnv, self).__init__()

        # Publishers
        self._goal_pub = rospy.Publisher('/robot_1/move_base/goal', MoveBaseActionGoal, queue_size = 1)
        self._pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size = 1)

        # Subscribers
        rospy.Subscriber("/robot_1/map", OccupancyGrid, self._map_data_callback)
        rospy.Subscriber("/robot_1/map_metadata", MapMetaData, self._map_metadata_callback)
        rospy.Subscriber("/robot_1/move_base_node/NavfnROS/plan", Path, self._path_callback)
        rospy.Subscriber("/robot_1/move_base/goal", MoveBaseActionGoal, self._goal_callback)
        rospy.Subscriber("/robot_1/odom", Odometry, self._pose_callback)
        rospy.Subscriber("/robot_1/mobile_base/events/bumper", BumperEvent, self._bumper_callback)
        rospy.Subscriber("/end_local_cond", String, self._local_end_callback)
        rospy.Subscriber("/end_global_cond", String, self._global_end_callback)

        self.start_signal = False
        self.map_data = []
        self.path_x = []
        self.path_y = []
        self.map_exist = False
        self.metadata_exist = False
        self.path_exist = False
        self.path_error = False
        self.pose = [0, 0, 0]
        self.map_metadata = MapMetaData()
        self.temp_x = 0
        self.temp_y = 0
        self.bump_times = 0
        self.bumper = -1
        self.local_end = False
        self.global_end = False
        self.search_signal = False

    def init_pose(self):
        """
        This will init the pose of robot
        """
        pose = self.get_pose()
        p = PoseWithCovarianceStamped()
        p.header.frame_id = "/robot_1/map"
        p.pose.pose.position.x = pose[0]
        p.pose.pose.position.y = pose[1]
        p.pose.pose.orientation.x = 0
        p.pose.pose.orientation.y = 0
        p.pose.pose.orientation.z = 0
        p.pose.pose.orientation.w = 0
        p.pose.covariance[6*0+0] = 0.5 * 0.5
        p.pose.covariance[6*1+1] = 0.5 * 0.5
        p.pose.covariance[6*3+3] = math.pi/12.0 * math.pi/12.0
        self._pose_pub.publish(p)

    def _local_end_callback(self, data):
        self.local_end = True

    def _global_end_callback(self, data):
        self.global_end = True

    def _bumper_callback(self, data):
        '''
        LEFT:0
        CENTER:1
        RIGHT:2
        LEFT_RECOVERY:1004
        CENTER_RECOVERY:1005
        RIGHT_RECOVERY:1006
        '''
        if data.bumper == 0:
            self.bumper = 0
        elif data.bumper == 1:
            self.bumper = 1
        elif data.bumper == 2:
            self.bumper = 2
        # self.bump_times = self.bump_times + 1

    def _map_data_callback(self, data):
        self.start_signal = True
        map_exist = self.get_map_exist()
        if map_exist == False:
            height = data.info.height
            width = data.info.width
            length = height * width
            map_data = numpy.empty(length)
            for i in range(height):
                for j in range(width):
                    index = j + (height - i - 1) * width
                    map_data[index] = data.data[index]
            self.map_data = []
            for index in range(length):
               self.map_data.append(map_data[index])
            self.map_exist = True

    def _map_metadata_callback(self, data):
        self.map_metadata = data
        self.metadata_exist = True
        self.start_signal = True

    def _path_callback(self, data):
        path_exist = self.get_path_exist()
        path_error = self.get_path_error()
        search_signal = self.get_search_signal()
        if path_exist == False and search_signal == True and len(data.header.frame_id) != 0:
            rospy.loginfo("Getting path")
            self.path_x = []
            self.path_y = []
            temp_x, temp_y = self.get_temp_goal()
            x = 0
            y = 0
            id = 0
            while int(abs(x - temp_x)) > 2 or int(abs(y - temp_y)) > 2:
                # print("x, y:", x, " ", y)
                # print("temp_x, temp_y:", temp_x, " ", temp_y)
                x = data.poses[id].pose.position.x
                y = data.poses[id].pose.position.y
                self.path_x.append(x)
                self.path_y.append(y)
                id = id + 1
            self.path_exist = True
            self.search_signal = False
        elif path_error == False and len(data.header.frame_id) == 0:
            pose_1 = self.get_pose()
            pose_2 = [0, 0, 0]
            times = 0
            while True:
                if times >= 10:
                    self.path_error = True
                    rospy.loginfo("PATH LOST")
                    break
                pose_2 = self.get_pose()
                if pose_1[0] == pose_2[0] and pose_1[1] == pose_2[1]:
                    times = times + 1
                else:
                    break
            # self._set_goal(0, 0, 997)

    def _goal_callback(self, data):
        if data.goal.target_pose.header.seq == 1000: # receive temp goal
            self.temp_x = data.goal.target_pose.pose.position.x
            self.temp_y = data.goal.target_pose.pose.position.y
            
    
    def _pose_callback(self, data):
        self.pose[0] = data.pose.pose.position.x
        self.pose[1] = data.pose.pose.position.y
        self.pose[2] = data.pose.pose.position.z

    def get_local_end(self):
        return self.local_end

    def get_global_end(self):
        return self.global_end

    def get_bump_times(self):
        return self.bump_times

    def get_bumper(self):
        return self.bumper

    def get_map_data(self):
        return self.map_data

    def get_map_metadata(self):
        return self.map_metadata

    def get_path(self):
        return self.path_x, self.path_y

    def get_pose(self):
        return self.pose

    def get_position(self):
        pose = self.get_pose()
        position = torch.zeros(1, 3).to(device)
        position[0][0] = pose[0]
        position[0][1] = pose[1]
        position[0][2] = pose[2]
        return position

    def get_start_signal(self):
        return self.start_signal

    def get_map_exist(self):
        return self.map_exist

    def get_metadata_exist(self):
        return self.metadata_exist

    def get_path_exist(self):
        return self.path_exist

    def get_path_error(self):
        return self.path_error

    def get_temp_goal(self):
        return self.temp_x, self.temp_y

    def get_search_signal(self):
        return self.search_signal

    def get_state(self):
        observation = self._get_observations()
        position = self.get_position()
        M = 481
        return {
            "global_map": observation,
            "global_pose": position,
            "collision_map": torch.zeros(1, M, M).to(device),
            "visited_map": torch.zeros(1, 1, M, M).to(device),
        }

    def _set_goal(self, x, y, seq):
        """
        This will set the goal of the turtlebot2
        seq = 997:SHUTDOWN
        seq = 1000:WAIT
        seq = 1001:GO
        seq = 1003:SPIN
        """
        # rospy.logdebug("Start Set Goal ==>"+str(x)+str(' ')+str(y))
        goal_value = MoveBaseActionGoal()
        goal_value.goal.target_pose.header.frame_id = "robot_1/map";
        goal_value.goal.target_pose.header.seq = seq;
        goal_value.goal.target_pose.pose.position.x = x;
        goal_value.goal.target_pose.pose.position.y = y;
        goal_value.goal.target_pose.pose.position.z = 0.0;
        goal_value.goal.target_pose.pose.orientation.x = 0.0;
        goal_value.goal.target_pose.pose.orientation.y = 0.0;
        goal_value.goal.target_pose.pose.orientation.z = 0.0;
        goal_value.goal.target_pose.pose.orientation.w = 1.0;       
        self._goal_pub.publish(goal_value)

    def _get_observations(self):
        """
        Here we get current observations of the turtlebot2
        """
        rospy.logdebug("Start Get Current Observations ==>")
        M = 481
        global_map = torch.rand(1, 2, M, M).to(device);
        for i in range(M):
          for j in range(M): # padded area
            global_map[0][0][i][j] = 1
            global_map[0][1][i][j] = 0
        while self.get_map_exist() == False:
            continue
        map_data = self.get_map_data()
        while self.get_metadata_exist() == False:
            continue
        map_metadata = self.get_map_metadata()
        origin_x = map_metadata.origin.position.x
        origin_y = map_metadata.origin.position.y
        map_H = int(map_metadata.height)
        map_W = int(map_metadata.width)
        length = len(map_data)
        while length != map_H * map_W:
            if length > map_H * map_W:
                rospy.loginfo("Reload map metadata")
                # map_metadata = self.get_map_metadata()
                # origin_x = map_metadata.origin.position.x
                # origin_y = map_metadata.origin.position.y
                # map_H = int(map_metadata.height)
                # map_W = int(map_metadata.width)
                gap = int(length - map_H * map_W)
                map_data = map_data[:length - gap]
                length = len(map_data)
            elif length < map_H * map_W:
                rospy.loginfo("Reload mapdata")
                # map_data = self.get_map_data()
                gap = int(map_H * map_W - length)
                for i in range(gap):
                    map_data.append(-1)
                length = len(map_data)
        map_data_ = numpy.reshape(map_data, (map_H,map_W))
        for i in range(map_H):
          for j in range(map_W):
            map_data_[i][j] = map_data[j+(map_H-i-1)*map_W]
        origin_x, origin_y = self.world2map(origin_x, origin_y, M, M, 0.1)
        H_d = int(origin_x - map_H)
        W_d = int(origin_y)
        for i in range(map_H):
          for j in range(map_W):
            if i + H_d < M and i + H_d >= 0 and j + W_d < M and j + W_d >= 0:
              if map_data_[i][j] == -1: # unexplored space
                global_map[0][0][i+H_d][j+W_d] = 1
                global_map[0][1][i+H_d][j+W_d] = 0
              elif map_data_[i][j] == 0: # free space
                global_map[0][0][i+H_d][j+W_d] = 0
                global_map[0][1][i+H_d][j+W_d] = 1
              elif map_data_[i][j] == 100: # obstacles
                global_map[0][0][i+H_d][j+W_d] = 1
                global_map[0][1][i+H_d][j+W_d] = 1
        rospy.logdebug("END Get Current Observations ==>")
        self.search_signal = True
        rospy.logdebug("Allow to search path within free space")
        return global_map

    def reset(self):
         """
         Here we reset the env
         """
         self.map_exist = False
         self.metadata_exist = False
         self.path_exist = False
         self.path_error = False
         self.bump_times = 0
         self.bumper = -1
         self.local_end = False
         self.search_signal = False
         # self.spin()

    def global_reset(self):
        """
        Here we reset the global area
        """
        self.global_end = False
        # self.init_pose()
        rospy.loginfo("INIT POSE")

    def bumper_reset(self):
        """
        Here we reset the bumper
        """
        self.bumper = -1

    def path_error_reset(self):
        self.path_error = False

    def spin(self):
        """
        Here we spin the robot
        """
        for i in range(10000):
             self._set_goal(0, 0, 1003)

    def bump_recovery(self, seq):
        """
        Here we save the robot from bumping
        """
        self._set_goal(0, 0, seq)

    def world2map(self, x_world, y_world, H, W, map_scale):
        """
        Here we convert world coordinates to map coordinates
        """
        Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
        Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2
        x_map = int(Hby2 - y_world / map_scale)
        y_map = int(Wby2 + x_world / map_scale)
        return x_map, y_map

    def _shut_down(self):
        """
        Here we publish the shutdown signal to close the current environment
        """
        rospy.loginfo("Shut down current env")
        goal_value = MoveBaseActionGoal()
        goal_value.goal.target_pose.header.frame_id = "robot_1/map";
        goal_value.goal.target_pose.header.seq = 997; # shutdown signal
        goal_value.goal.target_pose.pose.position.x = 0.0;
        goal_value.goal.target_pose.pose.position.y = 0.0;
        goal_value.goal.target_pose.pose.position.z = 0.0;
        goal_value.goal.target_pose.pose.orientation.x = 0.0;
        goal_value.goal.target_pose.pose.orientation.y = 0.0;
        goal_value.goal.target_pose.pose.orientation.z = 0.0;
        goal_value.goal.target_pose.pose.orientation.w = 1.0;       
        self._goal_pub.publish(goal_value)
        self.start_signal = False
