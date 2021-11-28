#!/usr/bin/env python

"""
    Moving Horizon Estimation Localization
    Copyright © 2020 Mostafa Osman

    Permission is hereby granted, free of charge, 
    to any person obtaining a copy of this software 
    and associated documentation files (the “Software”), 
    to deal in the Software without restriction, 
    including without limitation the rights to use, 
    copy, modify, merge, publish, distribute, sublicense, 
    and/or sell copies of the Software, and to permit persons to whom 
    the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included 
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import rospy
import math
import numpy as np
from tf.transformations import euler_from_quaternion

# ROS messages
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry


class OdometryInterface:
    def __init__(self):
        self._Id = 0
        self._topic_name = ''
        self._odom_config = []
        self._number_of_measurements = 0
        self._C_matrix = None
        self._measurement_memory = None
        self.first_call = True
        self._prev_measurement = None
        self._measurement = 0
        self.updated = False
        self.type = ""
        self._frame_id = ''
        self._timestamp = 0
        self._measurement_information = None
        self._measurement_covariance = None
        self.measurement_type = ''
        self.mahalanobis_threshold = 0

    def Odometry_initialize(self, Id, topic_name, odom_config, N_mhe, mahalanobis_threshold):
        # Initializing the Odometry with its topic name and Id number
        self._Id = Id
        self._topic_name = topic_name
        self._odom_config = odom_config
        self._number_of_measurements = int(np.sum(odom_config[0:3])) + int(np.sum(odom_config[6:15]))

        if odom_config[3] or odom_config[4] or odom_config[5]:
            self._number_of_measurements += 3

        self._C_matrix = np.zeros((self._number_of_measurements, 15))
        self.form_C_matrix()
        self._measurement_memory = np.zeros((self._number_of_measurements, N_mhe))
        self.first_call = True
        self._prev_measurement = np.zeros(self._number_of_measurements)
        self._measurement = np.zeros(self._number_of_measurements)
        self.updated = False

        self.mahalanobis_threshold = mahalanobis_threshold

        self.measurement_type = 'odom'

    def Imu_initialize(self, Id, topic_name, odom_config, N_mhe, mahalanobis_threshold):
        # Initializing the Odometry with its topic name and Id number
        self._Id = Id
        self._topic_name = topic_name
        self._odom_config = odom_config
        self._number_of_measurements = int(np.sum(odom_config))
        self._C_matrix = np.zeros((self._number_of_measurements, 15))
        self.form_C_matrix()
        self._measurement_memory = np.zeros((self._number_of_measurements, N_mhe))
        self.first_call = True
        self._prev_measurement = np.zeros(self._number_of_measurements)
        self._measurement = np.zeros(self._number_of_measurements)
        self.updated = False

        self.mahalanobis_threshold = mahalanobis_threshold

        self.measurement_type = 'imu'

    def form_C_matrix(self):
        C_matrix_dummy = np.zeros((15, 15))
        for i in range(0, 15):
            C_matrix_dummy[i, i] = self._odom_config[i]

        if self._odom_config[3] or self._odom_config[4] or self._odom_config[5]:
            C_matrix_dummy[3, 3] = 1.0
            C_matrix_dummy[4, 4] = 1.0
            C_matrix_dummy[5, 5] = 1.0

        k = 0
        for i in range(0, 15):
            if not np.sum(C_matrix_dummy[i, :]) == 0:
                self._C_matrix[k, :] = C_matrix_dummy[i, :]
                k += 1

    # Limit any angle from 0 to 360 degrees
    @staticmethod
    def limit_angles(unlimited_angle):
        return math.atan2(math.sin(unlimited_angle), math.cos(unlimited_angle))

    def odometryCb(self, data):
        # Reading the Odometry Data and pushing them to the data vectors
        # Making a tuple in order to be able to use the euler_from_quaternion function
        states = []
        if self.measurement_type == 'odom':
            quat = (data.pose.pose.orientation.x,
                    data.pose.pose.orientation.y,
                    data.pose.pose.orientation.z,
                    data.pose.pose.orientation.w)

            (roll, pitch, yaw) = euler_from_quaternion(quat)

            if self._odom_config[3] == 0:
                roll = 0

            if self._odom_config[4] == 0:
                pitch = 0

            if self._odom_config[5] == 0:
                yaw = 0

            # Need to build the measurement matrix using the odometry configuration
            states = [data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z,
                      self.limit_angles(roll), self.limit_angles(pitch), self.limit_angles(yaw),
                      data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z,
                      data.twist.twist.angular.x, data.twist.twist.angular.y, data.twist.twist.angular.z,
                      0, 0, 0]

        elif self.measurement_type == 'imu':
            quat = (data.orientation.x,
                    data.orientation.y,
                    data.orientation.z,
                    data.orientation.w)

            (roll, pitch, yaw) = euler_from_quaternion(quat)

            if self._odom_config[3] == 0:
                roll = 0

            if self._odom_config[4] == 0:
                pitch = 0

            if self._odom_config[5] == 0:
                yaw = 0

            # Need to build the measurement matrix using the odometry configuration
            states = [0, 0, 0,
                      self.limit_angles(roll), self.limit_angles(pitch), self.limit_angles(yaw),
                      0, 0, 0,
                      data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z,
                      data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]

        self._measurement = np.matmul(self._C_matrix, states)

        # Storing the parent frame_id and the timestamp
        self._frame_id = data.header.frame_id
        self._timestamp = data.header.stamp
        # self._timestamp = rospy.get_time()

        if self.measurement_type == 'odom':
            # Need to build the covariance matrix using the odometry configuration
            covariance = np.asarray(data.pose.covariance)
            covariance = np.reshape(covariance, (6, 6))

            velocity_covariance = np.asarray(data.twist.covariance)
            velocity_covariance = np.reshape(velocity_covariance, (6, 6))

            acceleration_covariance = np.identity(3)

            # covariance = np.concatenate((covariance, np.zeros((6, 6))), axis=0)
            # velocity_covariance = np.concatenate((np.zeros((6, 6)), velocity_covariance), axis=0)

            full_covariance = np.identity(15)
            full_covariance[0:6, 0:6] = covariance
            full_covariance[6:12, 6:12] = velocity_covariance
            full_covariance[12:15, 12:15] = acceleration_covariance

            # full_covariance = casadi.horzcat(covariance, velocity_covariance)
            # full_covariance = np.concatenate((covariance, velocity_covariance), axis=1)

            self._measurement_covariance = np.matmul(self._C_matrix, np.matmul(full_covariance,
                                                                               np.transpose(self._C_matrix)))
            #
            # for i in range(0, self._number_of_measurements):
            # 	for j in range(0, self._number_of_measurements):
            # 		if not i == j:
            # 			self._measurement_covariance[i, j] = 0

            # print self._measurement_covariance

        elif self.measurement_type == 'imu':
            # Need to build the covariance matrix using the odometry configuration
            orientation_covariance = np.asarray(data.orientation_covariance)
            orientation_covariance = np.reshape(orientation_covariance, (3, 3))

            angular_velocity_covariance = np.asarray(data.angular_velocity_covariance)
            angular_velocity_covariance = np.reshape(angular_velocity_covariance, (3, 3))

            acceleration_covariance = np.asarray(data.linear_acceleration_covariance)
            acceleration_covariance = np.reshape(acceleration_covariance, (3, 3))

            full_covariance = np.identity(15)
            full_covariance[3:6, 3:6] = orientation_covariance
            full_covariance[9:12, 9:12] = angular_velocity_covariance
            full_covariance[12:15, 12:15] = acceleration_covariance

            self._measurement_covariance = np.matmul(self._C_matrix, np.matmul(full_covariance,
                                                                               np.transpose(self._C_matrix)))

        for i in range(0, self._number_of_measurements):
            if self._measurement_covariance[i, i] < 1e-15:
                self._measurement_covariance[i, i] = 1e-4

        self._measurement_information = np.linalg.inv(self._measurement_covariance)

        self.updated = True

        # Change the first_cb flag
        self.first_call = False
