import os

import scipy
import scipy.sparse as sp
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph


def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def generate_partition(labels, ratio):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}  ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(idx)
            total_num -= 1
        else:
            p_unlabeled.append(idx)
    return p_labeled, p_unlabeled


def load_data(dataset, path="./data/", ):
    data = sio.loadmat(path + dataset + '.mat')
    features = data['X']
    feature_list = []

    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    for i in range(features.shape[1]):
        # features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if ss.isspmatrix_csr(feature):
            feature = feature.todense()
            print("sparse")
        feature_list.append(feature)
    return feature_list, labels

def adjs_to_Lap_single(adj, knns=10):
    # 遍历每个视图

    temp = sp.coo_matrix(adj)
    # build symmetric adjacency matrix
    temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
    lap=sparse_mx_to_torch_sparse_tensor(construct_laplacian(temp))
    return lap


def load_data_2(dataset, path="/data/dusd/code/Bi_sparse_co_clustering/_multiview datasets", ):
    data = sio.loadmat(path + dataset + '.mat')
    features = data['X']
    feature_list = []
    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    for i in range(features.shape[1]):
        features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if ss.isspmatrix_csr(feature):
            feature = feature.todense()
            print("sparse")
        feature_list.append(feature)
    return feature_list, labels

def load_data_single(dataset, path="./data/"):
    data = sio.loadmat(path + dataset + '.mat')
    features = data['X']
    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    adj=data['adj']
    features = normalize(features)
    if ss.isspmatrix_csr(features):
        features = features.todense()
        print("sparse")
    return features, labels,adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def features_to_Lap( dataset,features, device, knns=10):
    laps = []
    adjs = []
    for i in range(len(features)):
        direction_judge = './adj_matrix/' + dataset + '/' + 'v' + str(i) + '_knn' + str(knns) + '_adj.npz'
        if os.path.exists(direction_judge):
            print("Loading the adjacency matrix of " + str(i) + "th view of " + dataset)
            temp_adj = ss.load_npz(direction_judge)
            temp_lap=sp.eye(temp_adj.shape[0]) - temp_adj
            adj = torch.from_numpy(temp_adj.todense()).float().to(device)
            lap = torch.from_numpy(temp_lap.todense()).float().to(device)
            adjs.append(adj)
            laps.append(lap)

        else:
            print("Constructing the adjacency matrix of " + str(i) + "th view of " + dataset)

            # 返回该视图下每个样本距离最近的前n个样本
            temp = kneighbors_graph(features[i], knns, mode="distance")
            # 生成矩阵
            temp = sp.coo_matrix(temp)
            # build symmetric adjacency matrix
            temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)

            temp_adj = construct_adj_wave(temp)
            temp_lap = sp.eye(temp_adj.shape[0]) - temp_adj

            save_direction = './adj_matrix/' + dataset + '/'
            if not os.path.exists(save_direction):
                os.makedirs(save_direction)
            print("Saving the adjacency matrix to " + save_direction)
            ss.save_npz(save_direction + 'v' + str(i) + '_knn' + str(knns) + '_adj.npz', temp_adj)

            adj =sparse_mx_to_torch_sparse_tensor(temp_adj).float().to(device)
            lap = sparse_mx_to_torch_sparse_tensor(temp_lap).float().to(device)

            adjs.append(adj)
            laps.append(lap)

    return laps,adjs




def features_to_Lap_single(features, knns=10):
    lap = []
    sim = []

    temp = kneighbors_graph(features, knns, mode="distance")
    temp = sp.coo_matrix(temp)
    temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
    sim.append(sparse_mx_to_torch_sparse_tensor(temp))
    lap.append(sparse_mx_to_torch_sparse_tensor(construct_laplacian(temp)))
    return sim, lap

def features_to_Adj(features, knns=10):
    adj = []
    for i in range(len(features)):
        temp = kneighbors_graph(features[i], knns, mode="distance")
        temp = sp.coo_matrix(temp)
        temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
        adj.append(temp.tocoo().astype(np.float32))
    return adj

def construct_adj_wave(adj):
    """
        construct the Laplacian matrix
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj_ = sp.coo_matrix(adj)
    # adj_ = sp.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    # print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    # lp = sp.eye(adj.shape[0]) - adj_wave
    lp = adj_wave
    return lp

def getInitF(dataset, n_view, datasetGFW_dir="./datasetGFW"):
    data = scipy.io.loadmat(os.path.join(datasetGFW_dir, dataset))
    # print(data.keys())
    Z = data[dataset]
    Z_init = Z[0][0]
    for i in range(1, Z.shape[1]):
        Z_init += Z[0][i]
    return Z_init / n_view

def normalization(data):
    maxVal = torch.max(data)
    minVal = torch.min(data)
    data = (data - minVal)//(maxVal - minVal)
    return data


