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

# Importing Python and ROS libraries
import rospy
import math
import threading
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import threading

# Importing ROS messages
from nav_msgs.msg import Odometry

# Importing Casadi
from sys import path
import rospkg

rospack = rospkg.RosPack()
rospack.list()
casadiPath = rospack.get_path('mhe_localization')
casadiPath = casadiPath + '/libs/casadi-linux-py27-58aa427'
path.append(casadiPath)

from casadi import *


class mhe_localization:
    pub = rospy.Publisher("/fusion_odometry", Odometry, queue_size=1)

    def __init__(self, Odometries, freq, N_mhe, N_of_odometries, mahalabonis_thresholds, lock):
        self.Odometries = Odometries
        self._freq = freq
        self._delta_T = 1.0 / freq
        self._N_mhe = N_mhe
        self._N_of_odometries = N_of_odometries
        self._total_number_of_measured_states = 0
        self._index = 0
        self._delta_T_ = self._delta_T
        self.mahalabonis_thresholds = mahalabonis_thresholds
        self.propagated_states = None
        self.g = casadi.SX([])
        self.args = {}
        self.opts = {}
        self.P = None
        self.obj = 0
        self.arrival_obj = 0
        self.process_obj = 0

        self.lock = lock

        Q = np.zeros((15, 15))
        # Process Noise Covariance
        Q[0, 0] = 100.1
        Q[1, 1] = 100.1
        Q[2, 2] = 100.6
        Q[3, 3] = 100.3
        Q[4, 4] = 100.3
        Q[5, 5] = 100.6
        Q[6, 6] = 100.25
        Q[7, 7] = 100.25
        Q[8, 8] = 100.4
        Q[9, 9] = 100.1
        Q[10, 10] = 100.1
        Q[11, 11] = 100.2
        Q[12, 12] = 100.1
        Q[13, 13] = 100.1
        Q[14, 14] = 100.15

        self.Q_euler = Q

        self.Q = np.identity(21)
        rotation_covariance = self.compute_so3_covariance(Q[3:6, 3:6])

        self.Q[3:12, 3:12] = rotation_covariance
        self.Q[0:3, 0:3] = Q[0:3, 0:3]
        self.Q[12:18, 12:18] = Q[6:12, 6:12]
        self.Q[18:21, 18:21] = Q[12:15, 12:15]

        self.Q_inv = np.linalg.inv(self.Q)

        self.P_0 = np.identity(15)
        self.P_0[0, 0] = 1e-1
        self.P_0[1, 1] = 1e-1
        self.P_0[2, 2] = 1e-1
        self.P_0[3, 3] = 1e-1
        self.P_0[4, 4] = 1e-1
        self.P_0[5, 5] = 1e-1
        self.P_0[6, 6] = 1e-1
        self.P_0[7, 7] = 1e-1
        self.P_0[8, 8] = 1e-1
        self.P_0[9, 9] = 1e-1
        self.P_0[10, 10] = 1e-1
        self.P_0[11, 11] = 1e-1
        self.P_0[12, 12] = 1e-1
        self.P_0[13, 13] = 1e-1
        self.P_0[14, 14] = 1e-1

        # self.posterior_covariance[] = self.P_0
        self.posterior_information = np.linalg.inv(self.P_0)

        for i in range(0, N_of_odometries):
            self._total_number_of_measured_states += Odometries[i]._number_of_measurements

        self.X0 = np.zeros((15, N_mhe))  # The states memory to be used as an initial guess

        self.x_sol = np.zeros((15, 1))
        self.x_sol_static = np.zeros((15, 1))

        self._measurements_memory = [np.zeros(self._total_number_of_measured_states) for i in range(self._N_mhe)]
        self._information_memory = [
            np.zeros((self._total_number_of_measured_states, self._total_number_of_measured_states)) for i in
            range(0, self._N_mhe)]
        self._C_matrix_memory = [np.zeros((self._total_number_of_measured_states, 15)) for i in range(0, self._N_mhe)]
        self._T_ = [self._delta_T for i in range(0, self._N_mhe)]
        self.states_memory = [self.x_sol_static for i in range(0, self._N_mhe)]
        self.posterior_covariance = [self.P_0 for i in range(0, self._N_mhe)]
        self.prior_states = [np.zeros(15) for i in range(0, self._N_mhe)]

        # 1 -> translation
        # 2 -> rotation
        # 3 -> velocity or acceleration
        self.measurements_description = [np.zeros(self._total_number_of_measured_states) for i in range(self._N_mhe)]

        self.first_trial = True

        self.new_storage = False

        self.timestamp = rospy.get_rostime()

    @staticmethod
    def form_description(C_matrix):
        description = np.zeros(C_matrix.shape[0])
        for i in range(0, C_matrix.shape[0]):
            for j in range(0, C_matrix.shape[1]):
                if C_matrix[i, j] == 1.0:
                    if j == 0 or j == 1 or j == 2:
                        description[i] = 1
                    elif j == 3 or j == 4 or j == 5:
                        description[i] = 2
                    else:
                        description[i] = 3

        return description

    @staticmethod
    def subtract_angles(angle_k, angle_k_1):
        return math.atan2(math.sin(angle_k - angle_k_1), math.cos(angle_k - angle_k_1))

    # Limit any angle from 0 to 360 degrees
    @staticmethod
    def limit_angles(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def compute_so3_covariance(euler_covariance):
        roll = math.sqrt(euler_covariance[0, 0])
        pitch = math.sqrt(euler_covariance[1, 1])
        yaw = math.sqrt(euler_covariance[2, 2])

        sr = math.sin(roll)
        cr = math.cos(roll)

        sp = math.sin(pitch)
        cp = math.cos(pitch)

        sy = math.sin(yaw)
        cy = math.cos(yaw)

        R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]])

        R = np.reshape(R, 9)

        so3_covariance = np.identity(9)
        for i in range(0, 9):
            so3_covariance[i, i] = R[i] ** 2

        return so3_covariance

    @staticmethod
    def compute_so3_covariance_sym(euler_covariance):
        roll = casadi.sqrt(euler_covariance[0, 0])
        pitch = casadi.sqrt(euler_covariance[1, 1])
        yaw = casadi.sqrt(euler_covariance[2, 2])

        sr = casadi.sin(roll)
        cr = casadi.cos(roll)

        sp = casadi.sin(pitch)
        cp = casadi.cos(pitch)

        sy = casadi.sin(yaw)
        cy = casadi.cos(yaw)

        R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]])

        R = casadi.reshape(R, 9, 1)

        so3_covariance = np.identity(9)
        for i in range(0, 9):
            so3_covariance[i, i] = R[i] ** 2

        return so3_covariance

    @staticmethod
    def euler_angles_rotation(roll=0, pitch=0, yaw=0):
        sr = math.sin(roll)
        cr = math.cos(roll)

        sp = math.sin(pitch)
        cp = math.cos(pitch)

        sy = math.sin(yaw)
        cy = math.cos(yaw)

        R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]])

        return R

    @staticmethod
    def euler_angles_rotation_sym(roll, pitch, yaw):
        sr = casadi.sin(roll)
        cr = casadi.cos(roll)

        sp = casadi.sin(pitch)
        cp = casadi.cos(pitch)

        sy = casadi.sin(yaw)
        cy = casadi.cos(yaw)

        R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])

        return R

    def problem_formulation(self):
        x = casadi.SX.sym('x')
        y = casadi.SX.sym('y')
        z = casadi.SX.sym('z')

        vx = casadi.SX.sym('vx')
        vy = casadi.SX.sym('vy')
        vz = casadi.SX.sym('vz')

        roll = casadi.SX.sym('roll')
        pitch = casadi.SX.sym('pitch')
        yaw = casadi.SX.sym('yaw')

        wx = casadi.SX.sym('wx')
        wy = casadi.SX.sym('wy')
        wz = casadi.SX.sym('wz')

        ax = casadi.SX.sym('ax')
        ay = casadi.SX.sym('ay')
        az = casadi.SX.sym('az')

        self.states = casadi.vertcat(x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, ax, ay, az)
        self.n_states = self.states.size1()

        sr = casadi.sin(roll)
        cr = casadi.cos(roll)

        sp = casadi.sin(pitch)
        cp = casadi.cos(pitch)

        sy = casadi.sin(yaw)
        cy = casadi.cos(yaw)

        R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                           [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                           [-sp, cp * sr, cp * cr]])

        omni_directional_model = casadi.SX.zeros(self.n_states, self.n_states)
        omni_directional_model[0:6, 0:6] = casadi.SX.eye(6)
        omni_directional_model[0:3, 6:9] = self._delta_T * R
        omni_directional_model[0:3, 12:15] = ((self._delta_T * self._delta_T) / 2) * R
        omni_directional_model[3:6, 9:12] = self._delta_T * casadi.SX.eye(3)
        omni_directional_model[6:9, 6:9] = casadi.SX.eye(3)
        omni_directional_model[6:9, 12:15] = self._delta_T * casadi.SX.eye(3)
        omni_directional_model[9:12, 9:12] = casadi.SX.eye(3)
        omni_directional_model[12:15, 12:15] = casadi.SX.eye(3)

        # print omni_directional_model

        rhs = casadi.mtimes(omni_directional_model, self.states)

        self.g = casadi.SX([])

        self.f = casadi.Function('f', [self.states], [rhs])
        self.X = casadi.SX.sym('X', self.n_states, self._N_mhe)

        _jacobian = casadi.jacobian(rhs, self.states)
        self.evaluate_jacobian = casadi.Function('j', [self.states], [_jacobian])

        for k in range(0, self._N_mhe - 1):
            st = self.X[:, k]
            f_value = self.f(st)
            st_next = self.X[:, k + 1]

            contraint = f_value[0:15] - st_next[0:15]
            contraint[3] = casadi.atan2(casadi.sin(contraint[3]), casadi.cos(contraint[3]))
            contraint[4] = casadi.atan2(casadi.sin(contraint[4]), casadi.cos(contraint[4]))
            contraint[5] = casadi.atan2(casadi.sin(contraint[5]), casadi.cos(contraint[5]))

            # contraint[3] = casadi.cos(contraint[3])
            # contraint[4] = casadi.cos(contraint[4])
            # contraint[5] = casadi.cos(contraint[5])

            self.g = casadi.vertcat(self.g, contraint)

        ubg = 0
        lbg = 0

        # ############# Setting the Model Constraints ############################

        lbx = np.zeros((self.n_states * self._N_mhe, 1))
        ubx = np.zeros((self.n_states * self._N_mhe, 1))

        lbx[0: self.n_states * self._N_mhe] = -np.inf
        ubx[0: self.n_states * self._N_mhe] = np.inf

        lbx[6: self.n_states * self._N_mhe: self.n_states] = -np.inf
        ubx[6: self.n_states * self._N_mhe: self.n_states] = np.inf

        lbx[7: self.n_states * self._N_mhe: self.n_states] = -np.inf
        ubx[7: self.n_states * self._N_mhe: self.n_states] = np.inf

        lbx[11: self.n_states * self._N_mhe: self.n_states] = -np.inf
        ubx[11: self.n_states * self._N_mhe: self.n_states] = np.inf

        # ########################################################################

        self.args = {'lbx': lbx, 'ubx': ubx, 'ubg': ubg, 'lbg': lbg, 'x0': []}

        self.opts = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }

        self.P = casadi.SX.sym('P', self.n_states, self._N_mhe)

        # Formulating the Process Cost
        self.process_obj = 0
        for k in range(1, self._N_mhe):
            st = self.X[:, k]
            f_value = self.f(st)
            st_next = self.P[:, k]

            error_vector = st_next - f_value

            state_rotation = self.euler_angles_rotation_sym(f_value[3], f_value[4], f_value[5])
            measurement_rotation = self.euler_angles_rotation_sym(st_next[3], st_next[4], st_next[5])

            rotation_error = casadi.mtimes(measurement_rotation, state_rotation.T)
            error_vector[3] = rotation_error[2, 1] - rotation_error[1, 2]
            error_vector[4] = rotation_error[0, 2] - rotation_error[2, 0]
            error_vector[5] = rotation_error[1, 0] - rotation_error[0, 1]

            process_covariance = np.linalg.inv(self.Q_euler)

            self.process_obj = self.process_obj + casadi.mtimes(error_vector.T,
                                                                casadi.mtimes(process_covariance,
                                                                              error_vector))

        self.OPT_variables = casadi.reshape(self.X, self.n_states * self._N_mhe, 1)

    def prediction(self):
        sr = math.sin(self.x_sol[3])
        cr = math.cos(self.x_sol[3])

        sp = math.sin(self.x_sol[4])
        cp = math.cos(self.x_sol[4])

        sy = math.sin(self.x_sol[5])
        cy = math.cos(self.x_sol[5])

        R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]])

        omni_directional_model = np.zeros((self.n_states, self.n_states))
        omni_directional_model[0:6, 0:6] = np.identity(6)
        omni_directional_model[0:3, 6:9] = self._delta_T * R
        omni_directional_model[0:3, 12:15] = ((self._delta_T ** 2) / 2) * R
        omni_directional_model[3:6, 9:12] = self._delta_T * np.identity(3)
        omni_directional_model[6:9, 6:9] = np.identity(3)
        omni_directional_model[6:9, 12:15] = self._delta_T * np.identity(3)
        omni_directional_model[9:12, 9:12] = np.identity(3)
        omni_directional_model[12:15, 12:15] = np.identity(3)

        self.x_sol = np.reshape(self.x_sol, (15, 1))
        self.propagated_states = np.matmul(omni_directional_model, self.x_sol)

        self.propagated_states[3] = self.limit_angles(self.propagated_states[3])
        self.propagated_states[4] = self.limit_angles(self.propagated_states[4])
        self.propagated_states[5] = self.limit_angles(self.propagated_states[5])

    def measurements_storing(self):
        self.prediction()

        self._augmented_C_matrix = np.array([])
        self.measurements = np.array([])
        self.augmented_information = np.array([])

        for i in range(0, self._N_of_odometries):
            if not self.Odometries[i].first_call and self.Odometries[i].updated:
                self.new_storage = False
                self.Odometries[i].updated = False
                self.timestamp = self.Odometries[i]._timestamp
                predicted_states_in_measurement_space = np.matmul(self.Odometries[i]._C_matrix,
                                                                      self.propagated_states)

                innovation = predicted_states_in_measurement_space - self.Odometries[i]._measurement

                dummy_counter = 0
                for j in range(0, 12):
                    if self.Odometries[i]._odom_config[j]:
                        dummy_counter += 1
                        if j == 3 or j == 4 or j == 5:
                            innovation[dummy_counter - 1] = self.subtract_angles(
                                predicted_states_in_measurement_space[dummy_counter - 1],
                                self.Odometries[i]._measurement[dummy_counter - 1])

                mahalanobis_distance = np.matmul(innovation.T,
                                              np.matmul(self.Odometries[i]._measurement_information, innovation))

                if mahalanobis_distance < self.Odometries[i].mahalanobis_threshold:
                    if self._augmented_C_matrix.size == 0:
                        self._augmented_C_matrix = self.Odometries[i]._C_matrix
                    else:
                        self._augmented_C_matrix = np.concatenate((self._augmented_C_matrix,
                                                                   self.Odometries[i]._C_matrix), axis=0)

                    if self.measurements.size == 0:
                        self.measurements = self.Odometries[i]._measurement
                    else:
                        self.measurements = np.concatenate((self.measurements,
                                                       self.Odometries[i]._measurement), axis=0)

                    if self.augmented_information.size == 0:
                        self.augmented_information = self.Odometries[i]._measurement_information
                    else:
                        covariance_dummy = np.concatenate(
                            (np.zeros((self.augmented_information.shape[0], self.Odometries[i]._number_of_measurements)),
                            self.Odometries[i]._measurement_information), axis=0)
                        self.augmented_information = np.concatenate((self.augmented_information, np.zeros(
                            (self.Odometries[i]._number_of_measurements, self.augmented_information.shape[0]))), axis=0)
                        self.augmented_information = np.concatenate((self.augmented_information, covariance_dummy), axis=1)

        if not self.augmented_information.size == 0:
            # self.augmented_covariance = np.linalg.inv(self.augmented_covariance)
            self.description = self.form_description(self._augmented_C_matrix)

            self.lock.acquire()
            del self._measurements_memory[0]
            del self._information_memory[0]
            del self._C_matrix_memory[0]
            del self._T_[0]
            del self.prior_states[0]
            del self.measurements_description[0]

            self._measurements_memory.append(self.measurements)
            self._information_memory.append(self.augmented_information)
            self._C_matrix_memory.append(self._augmented_C_matrix)
            self._T_.append(self._delta_T_)
            self._delta_T_ = self._delta_T
            self.prior_states.append(self.propagated_states)
            self.measurements_description.append(self.description)

            self.new_storage = True

            self.lock.release()

            self.calculate_posterior_covariance()

            return True
        else:
            self.new_storage = False
            return False

    def calculate_posterior_covariance(self):
        # Calcualate the jacobian of the model
        jacobian = self.evaluate_jacobian(self.x_sol_static)

        riccati_1 = casadi.mtimes(jacobian, casadi.mtimes(self.posterior_covariance[self._N_mhe - 1], jacobian.T))
        covariance = np.linalg.inv(self._information_memory[self._N_mhe - 1])
        riccati_2 = covariance + casadi.mtimes(self._C_matrix_memory[self._N_mhe - 1],
                                        casadi.mtimes(self.posterior_covariance[self._N_mhe - 1],
                                               self._C_matrix_memory[self._N_mhe - 1].T))
        riccati_2_inv = np.linalg.inv(riccati_2)
        riccati_3 = casadi.mtimes(jacobian, casadi.mtimes(self.posterior_covariance[self._N_mhe - 1],
                                            casadi.mtimes(self._C_matrix_memory[self._N_mhe - 1].T,
                                                          casadi.mtimes(riccati_2_inv,
                                                                        casadi.mtimes(self._C_matrix_memory[self._N_mhe - 1],
                                                                                                        casadi.mtimes(
                                                                                                            self.posterior_covariance[
                                                                                                                self._N_mhe - 1],
                                                                                                            jacobian.T))))))

        posterior_covariance = self.Q_euler + riccati_1 - riccati_3

        del self.posterior_covariance[0]
        self.posterior_covariance.append(posterior_covariance)

    def mhe(self):
        if self.new_storage:
            self.new_storage = False

            self.obj = 0

            self.lock.acquire()
            for k in range(1, self._N_mhe):
                st = self.X[:, k]
                h_x = casadi.mtimes(self._C_matrix_memory[k], st)
                y_tilde = self._measurements_memory[k]

                # print y_tilde

                state_rotation_matrix = self.euler_angles_rotation_sym(st[3], st[4], st[5])

                if k == 0:
                    self.posterior_information = np.linalg.inv(self.posterior_covariance[0])
                    prior_rotation_matrix = self.euler_angles_rotation(self.states_memory[0][3],
                                                                       self.states_memory[0][4],
                                                                       self.states_memory[0][5])
                    error_vector = casadi.SX.zeros(15)
                    error_vector = self.states_memory[0] - st

                    rotation_error = casadi.mtimes(prior_rotation_matrix, state_rotation_matrix.T)
                    error_vector[3] = rotation_error[2, 1] - rotation_error[1, 2]
                    error_vector[4] = rotation_error[0, 2] - rotation_error[2, 0]
                    error_vector[5] = rotation_error[1, 0] - rotation_error[0, 1]

                    self.obj = self.obj + casadi.mtimes(error_vector.T, casadi.mtimes(self.posterior_information,
                                                                                      error_vector))
                else:
                    error_size = int(self.measurements_description[k].shape[0])
                    error_vector = casadi.SX.zeros(error_size)

                    buffer_counter = 1000
                    # Defining the error vector for step k
                    for z in range(0, self.measurements_description[k].shape[0]):
                        if self.measurements_description[k][z] == 1 or self.measurements_description[k][z] == 3:
                            # print self.measurements_description[k][z]
                            # print z
                            error_vector[z] = y_tilde[z] - h_x[z]

                        elif self.measurements_description[k][z] == 2 and not z == buffer_counter + 1 \
                                and not z == buffer_counter + 2:

                            rotation_angles = y_tilde[z: z+3]
                            buffer_counter = z

                            measurement_rotation = self.euler_angles_rotation(rotation_angles[0],
                                                                              rotation_angles[1],
                                                                              rotation_angles[2])

                            rotation_error = casadi.mtimes(measurement_rotation, state_rotation_matrix.T)
                            error_vector[z] = rotation_error[2, 1] - rotation_error[1, 2]
                            error_vector[z+1] = rotation_error[0, 2] - rotation_error[2, 0]
                            error_vector[z+2] = rotation_error[1, 0] - rotation_error[0, 1]

                    self.obj = self.obj + casadi.mtimes(error_vector.T, casadi.mtimes(self._information_memory[k],
                                                                                      error_vector))

            self.obj = self.obj + self.process_obj

            p = np.zeros((self.n_states, self._N_mhe))

            for i in range(0, self._N_mhe):
                p[:, i] = np.reshape(self.prior_states[i], 15)

            self.lock.release()

            self.args['x0'] = casadi.reshape(self.X0, self.n_states * self._N_mhe, 1)

            nlp_prob = {'f': self.obj, 'x': self.OPT_variables, 'g': self.g, 'p': self.P}

            self.solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, self.opts)

            sol = self.solver(x0=self.args['x0'], lbx=self.args['lbx'], ubx=self.args['ubx'], ubg=self.args['ubg'],
                              lbg=self.args['lbg'], p=p)

            X_sol = casadi.reshape(sol['x'], self.n_states, self._N_mhe)

            self.x_sol = X_sol[:, self._N_mhe - 1]

            self.publish_estimation()

            self.X0 = casadi.horzcat(X_sol[:, 1:], X_sol[:, -1])

            self.states_memory.append(self.x_sol)
            del self.states_memory[0]

    def publish_estimation(self):
        quat = quaternion_from_euler(self.x_sol[3], self.x_sol[4], self.x_sol[5])

        odom = Odometry()
        odom.header.stamp = self.timestamp

        odom.pose.pose.position.x = self.x_sol[0]
        odom.pose.pose.position.y = self.x_sol[1]
        odom.pose.pose.position.z = self.x_sol[2]

        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        odom.twist.twist.linear.x = self.x_sol[6]
        odom.twist.twist.linear.y = self.x_sol[7]
        odom.twist.twist.linear.z = self.x_sol[8]

        odom.twist.twist.angular.x = self.x_sol[9]
        odom.twist.twist.angular.y = self.x_sol[10]
        odom.twist.twist.angular.z = self.x_sol[11]

        odom.pose.covariance = np.asarray(casadi.reshape(self.posterior_covariance[self._N_mhe - 1][0:6, 0:6].T,
                                                         (36, 1)))
        odom.twist.covariance = np.asarray(
            casadi.reshape(self.posterior_covariance[self._N_mhe - 1][6:12, 6:12].T, (36, 1)))

        mhe_localization.pub.publish(odom)
