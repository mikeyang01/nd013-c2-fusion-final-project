# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        # reference: Exercise
        N = len(track_list) # N tracks
        M = len(meas_list) # M measurements        

        # the following only works for at most one track and one measurement
        self.unassigned_tracks = list(range(N))
        self.unassigned_meas = list(range(M))

        # initialize association matrix
        self.association_matrix = np.inf*np.ones((N,M)) 

        # loop over all tracks and all measurements to set up association matrix
        for i in range(N): 
            track = track_list[i]
            for j in range(M):
                meas = meas_list[j]
                dist = self.MHD(track, meas, KF)
                if self.gating(dist, meas.sensor):
                    self.association_matrix[i,j] = dist                
        # if len(meas_list) > 0:
        #    self.unassigned_meas = [0]
        # if len(track_list) > 0:
        #    self.unassigned_tracks = [0]
        # if len(meas_list) > 0 and len(track_list) > 0: 
        #    self.association_matrix = np.matrix([[0]])        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement: 最近的跟踪和测量
        # - find minimum entry in association matrix 找到关联矩阵中的最小条目
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############
        # the following only works for at most one track and one measurement
        # update_track = 0
        # update_meas = 0
        # reference: Exercise
                
        # find closest track and measurement for next update
        M = self.association_matrix
        if np.min(M) == np.inf:
            return np.nan, np.nan
               
        # get indices(index) of minimum entry
        index_min = np.unravel_index(np.argmin(M, axis=None), M.shape) 
        ind_track = index_min[0]
        ind_meas = index_min[1]
          
        # update association_matrix
        M = np.delete(M, ind_track, axis = 0)    
        M = np.delete(M, ind_meas, axis = 1)    
        self.association_matrix = M

        # update track_list and meas_list
        update_track = self.unassigned_tracks[ind_track]
        update_meas = self.unassigned_meas[ind_meas]
        
        # remove from list
        self.unassigned_tracks.remove(update_track)
        self.unassigned_meas.remove(update_meas)       
        ############
        # END student code
        ############ 
        
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        threshold = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
        if MHD < threshold:
            return True
        else:
            return False               
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############        
        # calc Mahalanobis distance
        # 马氏距离（Mahalanobis distance）：一个可以描述两个向量之间距离的度量，其考虑了两者的“位置”相近程度、不确定性、相关性。
        # H = np.matrix([[1, 0, 0, 0],[0, 1, 0, 0]]) 
        H = meas.sensor.get_H(track.x)
        # get_hx is to calculate nonlinear measurement expectation value h(x)   
        # get_H is to calculate Jacobian H at current x from h(x), 雅可比行列式是坐标变换理论的基础之一
        gamma = meas.z - meas.sensor.get_hx(track.x)
        #gamma = meas.z - H*track.x

        S = H*track.P*H.transpose() + meas.R
        MHD = gamma.transpose()*np.linalg.inv(S)*gamma # Mahalanobis distance formula
        return MHD            
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)