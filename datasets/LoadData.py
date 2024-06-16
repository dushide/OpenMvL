'''
Author: Luying Zhong
Date: 2022-05-30 14:53:30
LastEditTime: 2020-09-04 14:53:30
LastEditors: Luying Zhong
Description: Data loading tools
'''

import scipy.io
import os
import numpy as np
import h5py

### Load feature and ground truth data with .mat
def loadMatData(data_root,data_name):
    tmp = []

    if data_name=='Pubmed':
        data=h5py.File(data_root + data_name + ".mat")
        features=[]
        features.append(data['X1'][:].T)
        features.append(data['X2'][:].T)
        # features=np.array(features_tmp).reshape(-1, 1)
        gnd=data['Y'][:]
    else:
        data = scipy.io.loadmat(data_root + data_name + ".mat")
        features = data['X']

        gnd = data['Y']
    gnd = gnd.flatten()
    if gnd.min() == 1:
        gnd = gnd - 1
    return features, gnd

### Load similarity data with .mat
def loadSIM(data_name):
    data = scipy.io.loadmat(data_name)
    similaritis = data['W']
    return similaritis

def loadData_combinedWeight(dataset_name):
    '''
    Load multi-view data with a pre-calculated combined weight for all views.
    (Weight is computed by matlab codes)
    '''
    features, gnd = loadMatData(os.path.join("data",dataset_name+".mat"))
    W = loadSIM(os.path.join("data",dataset_name+"W.mat"))
    return features, gnd, W




