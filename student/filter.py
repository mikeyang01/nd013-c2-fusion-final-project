# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F       
        ############
        # F代表状态转移矩阵
        # system matrix
        dt = params.dt
        return np.matrix([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############        
        # Q代表状态噪声协方差矩阵
        # process noise covariance Q
        q = params.q
        dt = params.dt
        q1 = ((dt**3)/3) * q 
        q2 = ((dt**2)/2) * q 
        q3 = dt * q 
        return np.matrix([
                [q3, 0, 0, q2, 0, 0],
                [0, q3, 0, 0, q2, 0],
                [0, 0, q3, 0, 0, q2],
                [q2, 0, 0, q1, 0, 0],
                [0, q2, 0, 0, q1, 0],
                [0, 0, q2, 0, 0, q1]
            ])        
        ############
        # END student code
        ############        
        
    def predict(self, track):
        ############
        # TODO Step 
        # 1: predict state x and estimation error covariance P to next timestep, 
        # save x and P in track
        ############
        F = self.F()
        Q = self.Q()        
        x_ = F * track.x # state prediction
        P_ = F * track.P * F.transpose() + Q # covariance prediction
        track.set_x(x_)
        track.set_P(P_)
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 
        # 1: update state x and covariance P with associated measurement, 
        # save x and P in track
        ############        
        H = meas.sensor.get_H(track.x)
        gamma = self.gamma(track,meas) # residual
        S = self.S(track, meas, H) # covariance of residual
        P = track.P 
        x = track.x
        K = P*H.transpose()*np.linalg.inv(S) # Kalman gain
        x = x + K*gamma # state update
        I = np.identity(params.dim_state)
        P = (I - K*H) * P # covariance update
        track.set_x(x)
        track.set_P(P)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        # 在扩展卡尔曼滤波器（EKf）中，没有常量加到状态转移方程中，常常会出现状态量不足的情况，此时可以引入一个增益矩阵Gamma，它用来补充不足的状态量，同时保证系统的稳定。
        # Gamma矩阵是一个大小为(m,n)的矩阵，其中m是状态向量的维度，n是增益矩阵的维度。在EKf中，Gamma矩阵一般由扩展卡尔曼滤波器算法自动计算得出。
        # reference: udacity/nd013-c2-fusion-exercises
        gamma =  meas.z - meas.sensor.get_hx(track.x)
        return gamma        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        # S代表残差协方差矩阵，它是观测量与预测量之间误差的协方差矩阵。
        # 在状态更新的过程中，EKf算法通过计算残差协方差矩阵来评估观测量的可靠性，以决定是否应该更新状态估计。
        # 如果S矩阵越小，说明观测值与预测值之间的误差越小，可以更可靠地更新状态，反之则需要更小的增益值或更多的迭代来改善状态估计的准确性。
        H = meas.sensor.get_H(track.x)  
        P = track.P
        R = meas.R
        S = H*P*H.transpose() + R # covariance of residual       
        return S        
        ############
        # END student code
        ############ 