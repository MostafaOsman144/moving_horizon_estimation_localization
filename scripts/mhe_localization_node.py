#!/usr/bin/env python

"""
    Moving Horizon Estimation Localization
    Copyright © 2020 <copyright holders>

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

# Importing the classes for the node
import Odometry_class
import mhe_localization_class

# Importing Python and ROS libraries
import rospy
import math
import threading
import numpy as np
from tf.transformations import euler_from_quaternion

# Importing ROS messages
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

# Importing Casadi
from sys import path


global data_available
global MHE
global freq
global Odometries
global Odometries_topics
global N_mhe
global N_of_odometries
global N_of_imus
global mahalabonis_thresholds_odom
global first_interface

lock = threading.Lock()

def mhe_interface():
    global MHE
    global data_available
    global lock

    data_available = False
    first_run = True

    MHE = mhe_localization_class.mhe_localization(Odometries, freq, N_mhe, N_of_odometries + N_of_imus,
                                                  mahalabonis_thresholds_odom, lock)
    MHE.problem_formulation()

    rate = rospy.Rate(freq)

    while not rospy.is_shutdown():
        if first_run:
            for i in range(0, N_of_odometries + N_of_imus):
                if Odometries[i].first_call:
                    pass
                else:
                    first_run = False

            data_available = MHE.measurements_storing()
            rate.sleep()

        if not first_run:
            data_available = MHE.measurements_storing()
            rate.sleep()


def mhe_localization_node():
    global data_available
    global lock

    data_available = False

    starting_counter = 0
    rate = rospy.Rate(freq)  # hz

    while not rospy.is_shutdown():
        # mhe localizaiton
        if data_available:
            MHE.mhe()
            data_available = False

        rate.sleep()


def main():
    global freq
    global Odometries
    global Odometries_topics
    global N_mhe
    global N_of_odometries
    global N_of_imus
    global mahalabonis_thresholds_odom
    global first_interface

    # Initializing the ROS node with a name "mhe_localization"
    rospy.init_node('mhe_localization', anonymous=True)

    # Getting the parameters from the config file of the package (Number of Oodmetries and Odometries topics names)
    if rospy.has_param('N_of_Odometries'):
        N_of_odometries = rospy.get_param('N_of_Odometries')
    else:
        rospy.logerr('The parameter N_of_Odometries is not specified in the yaml file')

    if rospy.has_param('N_of_imus'):
        N_of_imus = rospy.get_param('N_of_imus')
    else:
        rospy.logerr('The parameter N_of_imus is not specified in the yaml file')

    if rospy.has_param('node_frequency'):
        freq = rospy.get_param('node_frequency')
    else:
        rospy.logerr('The parameter Sampling time is not specified in the yaml file')

    if rospy.has_param('estimation_horizon'):
        N_mhe = rospy.get_param('estimation_horizon')
    else:
        rospy.logerr('The parameter Sampling time is not specified in the yaml file')

    mahalabonis_thresholds_odom = np.zeros(N_of_odometries)
    mahalabonis_thresholds_imu = np.zeros(N_of_imus)
    Odometries_topics = [""] * N_of_odometries
    Imus_topics = [""] * N_of_imus
    Odometries_configuration = np.zeros((N_of_odometries, 15))
    Imu_configuration = np.zeros((N_of_imus, 15))
    Odometries = [Odometry_class.OdometryInterface() for i in range(N_of_odometries + N_of_imus)]

    # Initializing the Odometry Objects and subscribing to the topics indicated in the parameter file
    for i in range(0, N_of_odometries):
        if rospy.has_param('odom_' + str(i) + '_mahalanobis_threshold'):
            mahalabonis_thresholds_odom[i] = rospy.get_param("odom_" + str(i) + '_mahalanobis_threshold')
        else:
            rospy.logerr('Number of Odometries indicated in the yaml file is not correct')

    for i in range(0, N_of_imus):
        if rospy.has_param('imu_' + str(i) + '_mahalanobis_threshold'):
            mahalabonis_thresholds_imu[i] = rospy.get_param('imu_' + str(i) + '_mahalanobis_threshold')
        else:
            rospy.logerr('Number of imus indicated in the yaml file is not correct')

    for i in range(0, N_of_odometries):
        if rospy.has_param('odom_' + str(i)):
            Odometries_topics[i] = rospy.get_param("odom_" + str(i))
        else:
            rospy.logerr('Number of Odometries indicated in the yaml file is not correct')

    for i in range(0, N_of_imus):
        if rospy.has_param('imu_' + str(i)):
            Imus_topics[i] = rospy.get_param('imu_' + str(i))
        else:
            rospy.logerr('Number of imus indicated in the yaml file is not correct')

    for i in range(0, N_of_odometries):
        if rospy.has_param('odom_' + str(i) + '_config'):
            Odometries_configuration[i, :] = rospy.get_param("odom_" + str(i) + "_config")
        else:
            rospy.logerr('Number of Odometries indicated in the yaml file is not correct')

    for i in range(0, N_of_imus):
        if rospy.has_param('imu_' + str(i) + '_config'):
            Imu_configuration[i, :] = rospy.get_param('imu_' + str(i) + '_config')
        else:
            rospy.logerr('Number of imus indicated in the yaml file is not correct')

    for i in range(0, N_of_odometries):
        Odometries[i].Odometry_initialize(i, Odometries_topics[i], Odometries_configuration[i, :], N_mhe,
                                          mahalabonis_thresholds_odom[i])
        rospy.Subscriber(Odometries_topics[i], Odometry, Odometries[i].odometryCb, queue_size=1)

    for i in range(N_of_odometries, N_of_odometries + N_of_imus):
        Odometries[i].Imu_initialize(i, Imus_topics[i - N_of_odometries],
                                     Imu_configuration[i - N_of_odometries, :], N_mhe,
                                     mahalabonis_thresholds_imu[i - N_of_odometries])
        rospy.Subscriber(Imus_topics[i - N_of_odometries], Imu, Odometries[i].odometryCb, queue_size=1)

    threading.Thread(target=mhe_interface).start()
    threading.Thread(target=mhe_localization_node).start()
    
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
