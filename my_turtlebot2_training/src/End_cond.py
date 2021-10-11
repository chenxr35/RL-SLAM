#!/usr/bin/env python
# coding: utf-8
import rospy
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PolygonStamped
from move_base_msgs.msg import MoveBaseActionFeedback
from move_base_msgs.msg import MoveBaseActionGoal
import numpy as np
import csv
import message_filters
from math import *


class EndCond:

	def __init__(self):

		# data_file = open("database.csv", "a")
		# sscsv_writer = csv.writer(data_file, dialect='excel')
		# 576*576 - 28.8 -> 1728 * 1728 - 28.8*3

		"""
		Init method
		"""
		rospy.init_node('Input_maps', anonymous=True)

		# static configuration
		self.__width_total = 481
		self.__height_total = 481
		self.__cell_size = 0.05
		self.__init_robotx = 0   # m
		self.__init_roboty = 0   # m, not pixel

		self.__init_xpixel = 0   # first pixel of localmap
		self.__init_ypixel = 0
		self.__width_localmap = 0
		self.__height_localmap = 0
		self.__getgoal = False
		self.__last_state = []

		self.__endsearch_localfrontier = []
		self.__endsearch_globalfrontier = []

		# robot_size = RobotSize()
		self.__robot_size = 0.455 // 0.05   # 9 pixels
		self.__end_localpub = rospy.Publisher("/end_local_cond", String, queue_size=10)
		self.__end_globalpub   = rospy.Publisher("/end_global_cond", String, queue_size=10)


		self.create_msg_srv()


	def create_msg_srv(self):

		"""
		Defines publishers, subscribers, services and actions
		"""
		# rospy.Subscriber("/odom", Odometry, self.trajectory_robotposmapCb, queue_size=2, buff_size=52428800)

		rospy.Subscriber("/robot_1/map", OccupancyGrid, self.MapsCb)
		rospy.Subscriber("/robot_1/move_base/goal", MoveBaseActionGoal, self.GoalmapCb)

		# while not rospy.is_shutdown():
		rospy.loginfo("[START GETTING MAPS]")


	def GoalmapCb(self, FBEGoal):
		self.__getgoal = True


	def MapsCb(self, globalmapCb):

		if self.__getgoal:

			# self.__mapmsg.width = self.__width_total
			# self.__mapmsg.height = self.__height_total

			# params for frontier map and gloabl map
			self.__width_localmap = globalmapCb.info.width
			self.__height_localmap = globalmapCb.info.height

			pixel_xorigin = globalmapCb.info.origin.position.x
			pixel_yorigin = globalmapCb.info.origin.position.y
		
			# print("pixel_xorigin:", pixel_xorigin)
			# print("pixel_yorigin:", pixel_yorigin)
			# print("width_localmap:", width_localmap)
			# print("height_localmap:", height_localmap)

			self.__init_xpixel = int((self.__width_total*self.__cell_size/2 - (self.__init_robotx - pixel_xorigin)) // self.__cell_size)
			self.__init_ypixel = int((self.__height_total*self.__cell_size/2 - (self.__height_localmap*self.__cell_size - (self.__init_roboty - pixel_yorigin))) // self.__cell_size)

			# print("1__init_xpixel:", self.__init_xpixel)   184
			# print("1__init_ypixel:", self.__init_ypixel)   199

			# params of target_pose (FBE)
			"""
			goalx = FBEGoal.goal.target_pose.pose.position.x
			goaly = FBEGoal.goal.target_pose.pose.position.y
			x_goal_pixel = int((self.__width_localmap * self.__cell_size / 2 + goalx) // self.__cell_size)
			y_goal_pixel = int((self.__height_localmap * self.__cell_size / 2 - goaly) // self.__cell_size)
			"""


			globalcosmap = np.full((self.__height_total, self.__width_total), -1, dtype=int)
			frontier_localmap = np.full((self.__height_localmap, self.__width_localmap), -1, dtype=int)
			frontier_map = np.full((self.__height_total, self.__width_total), -1, dtype=int)
			goal_localmap = np.full((self.__height_localmap, self.__width_localmap), -1, dtype=int)
			goal_map = np.full((self.__height_total, self.__width_total), -1, dtype=int)	

			# print("x_robot_cell:", x_robot_cell)
			# print("y_robot_cell:", y_robot_cell)

			print("------------")

			# get globalmap
			# print(np.sum(np.array(globalmapCb.data)==0))
			for k in range(self.__height_localmap):
				frontier_localmap[self.__height_localmap-1-k][:] = globalmapCb.data[k*self.__width_localmap:(k*self.__width_localmap+self.__width_localmap)]


			# get frontier map
			mark = -2

			for row in range(len(frontier_localmap)):
				for column in range(len(frontier_localmap[0])):
					# print(globalcosmap[row][column] == 0)
					if frontier_localmap[row][column] == 0 and self.DeterPoint(frontier_localmap, row, column):
						frontier_localmap[row][column] = mark
						frontier_localmap = self.FBE(frontier_localmap, row, column, mark)
						mark -= 1


			score = {}
			score_small = []
			globalcosmap_1 = []
			for a in frontier_localmap:
				globalcosmap_1 += set(a)
			globalcosmap_1 = set(globalcosmap_1)

			# filter some short frontier
			for item in set(globalcosmap_1):

				# print("[%d]:%d"%(item, np.sum(np.array(globalcosmap)==item)))
			
				if item < -1 and np.sum(np.array(frontier_localmap)==item) > self.__robot_size:
					score[item] = np.sum(np.array(frontier_localmap)==item)

				elif item < -1 and np.sum(np.array(frontier_localmap)==item) <= self.__robot_size:
					score_small.append(item)
					# print(np.sum(np.array(frontier_localmap)==item))

			for i in range(len(frontier_localmap)):
				for j in range(len(frontier_localmap[i])):
					if frontier_localmap[i][j] in score_small:
						frontier_localmap[i][j] = -1


			# frontier has sequence
			# print("score:", score)
			self.__endsearch_globalfrontier.append(score)
			self.__endsearch_localfrontier.append(score)

			print(score)

			tmp_end_cond = []
			tmp_end_cond.append(score)

			# End of search
			print("self.__endsearch_globalfrontier:", self.__endsearch_globalfrontier)
			print("self.__endsearch_localfrontier:", self.__endsearch_localfrontier)

			if not score:
				rospy.loginfo("All map has been detected, wait for command!")
				self.__end_globalpub.publish("True")
				
			elif self.__last_state != score and self.end_localmap(self.__endsearch_localfrontier):
				rospy.loginfo("Local map has been detected, wait for next exploration!")
				self.__end_localpub.publish("True")
			'''
			else:
				self.__end_globalpub.publish("False")
				self.__end_localpub.publish("False")
				rospy.loginfo("Not an end")
			'''
			self.__getgoal = False

			"""
			print("self.__mapmsg.robotmap:", self.__mapmsg.robotmap)
			print("self.__mapmsg.trajectorymap:", self.__mapmsg.trajectorymap)
			print("self.__mapmsg.globalmap:", self.__mapmsg.globalmap)
			print("self.__mapmsg.frontiermap:", self.__mapmsg.fontiermap)
			print("self.__mapmsg.goalmap:", self.__mapmsg.goalmap)
			"""
			# self.get_dataset(self.__mapmsg.globalmap, self.__mapmsg.frontiermap, self.__mapmsg.robotmap, self.__mapmsg.trajectorymap, self.__mapmsg.goalmap)


	def end_fbe(self, dic_frontier, new_frontier):

		# print(np.sum(np.array(dic_frontier)==dic_frontier[-1]))
		if len(dic_frontier) == 0:
			rospy.loginfo("All map has been detected, wait for command!")
			return "globalend_true"

		elif len(dic_frontier) != 0 and new_frontier == 0:
			rospy.loginfo("All map has been detected, wait for command!")
			return "globalend_true"

		elif len(dic_frontier) >= 3 and dic_frontier[-1] == dic_frontier[-2] and np.sum(np.array(dic_frontier)==dic_frontier[-1])>=6:
			rospy.loginfo("All map has been detected, wait for command!")
			return "globalend_true"
			

	def end_localmap(self, dic_frontier):

		if len(dic_frontier) >= 2 and dic_frontier[-1] == dic_frontier[-2]:
			self.__last_state.append(dic_frontier[-1])
			self.__endsearch_localfrontier = []
			return True

		return False


	def get_dataset(self, globalmap, frontiermap, robotmap, trajectorymap, goalmap):

		self.__data_globalmap.writerow(globalmap)
		self.__data_frontiermap.writerow(frontiermap)
		self.__data_robotposmap.writerow(robotmap)
		self.__data_trajectorymap.writerow(trajectorymap)
		self.__data_goalmap.writerow(goalmap)


	def FBE(self, map, row, column, mark):

		for i in [row-1, row, row+1]:
			for j in [column-1, column, column+1]:
				if map[i][j] == 0 and self.DeterPoint(map, i, j):
					map[i][j] = mark
					map = self.FBE(map, i, j, mark)
		return map
		

	def DeterPoint(self, map, row, column):

		for i in [row-1, row, row+1]:
			for j in [column-1, column, column+1]:
				if map[i][j] == -1:
					return True
		return False


	def RobotSize(self):
		
		rospy.Subscriber("/move_base/global_costmap/footprint", PolygonStamped, Get_RobotSize)


	def Get_RobotSize(self, data):
		x_robot = []
		y_robot = []
		print(data.polygon.points)   # (16,)

		for item in data.polygon.points:
			x_robot.append(item.x)
			y_robot.append(item.y)
		print("x_robot:",x_robot)
		print("y_robot:", y_robot)
		x_size = max(x_robot) - min(x_robot)
		y_size = max(y_robot) - min(y_robot)

		print("x_size", x_size)
		print("y_size", y_size)
		print("max", max(x_size, y_size))
		return max(x_size, y_size)
		# x_robot.append(data.polygon.points.x)

if __name__ == '__main__':

	EndCond()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("over!")

