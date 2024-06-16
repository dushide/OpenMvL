import scipy.sparse as sp
import torch
import numpy as np
import scipy.io as scio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics



def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    F1_macro = metrics.f1_score(labels_true, labels_pred, average='macro')
    F1_micro = metrics.f1_score(labels_true, labels_pred, average='micro')
    return ACC, F1_macro, F1_micro

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def features_to_Lap(features,knns=10):

    # 遍历每个视图
    lap=[]
    for i in range(len(features)):
        # 返回该视图下每个样本距离最近的前n个样本
        temp = kneighbors_graph(features[i], knns, mode="distance")
        # 生成矩阵
        temp = sp.coo_matrix(temp)
        # 生成邻接矩阵
        temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
        lap.append(sparse_mx_to_torch_sparse_tensor(construct_laplacian(temp)))
    return lap

def features_to_Lap2(features,knns=10):

    # 遍历每个视图
    lap=[]
    sim=[]
    for i in range(len(features)):
        # 返回该视图下每个样本距离最近的前n个样本
        temp = kneighbors_graph(features[i], knns, mode="distance")
        # 生成矩阵
        temp = sp.coo_matrix(temp)
        # 生成邻接矩阵
        temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
        sim.append(sparse_mx_to_torch_sparse_tensor(temp))
        lap.append(sparse_mx_to_torch_sparse_tensor(construct_laplacian(temp)))
    return sim,lap

def features_to_adj(datasets,path="./data/", ):
    print("loading {} data...".format(datasets))
    data = scio.loadmat("{}{}.mat".format(path, datasets))
    x = data["X"]
    adj = []
    features = []
    nfeats=[]
    # 遍历每个视图
    for i in range(x.shape[1]):
        features.append(x[0, i])
        nfeats.append(len(features[i][1]))
        # 返回该视图下每个样本距离最近的前10个样本
        temp = kneighbors_graph(features[i], 10)
        # 生成矩阵
        temp = sp.coo_matrix(temp)
        # 生成邻接矩阵
        temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
        # 对应公式： A = A + I
        # normalize(A)对应公式：D^{-1/2} A \D^{-12}
        # temp = normalize(temp + sp.eye(temp.shape[0]))
        temp = construct_laplacian(temp)
        temp = sparse_mx_to_torch_sparse_tensor(temp)
        adj.append(temp)
    labels = data["Y"]
    labels = labels.reshape(-1, )
    return adj, features, labels, nfeats,len(nfeats),len(set(np.array(labels)))

def construct_laplacian(adj):
    """
        construct the Laplacian matrix
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj_ = sp.coo_matrix(adj)
    # adj_ = sp.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1)) # <class 'numpy.ndarray'> (n_samples, 1)
    # print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    # lp = sp.eye(adj.shape[0]) - adj_wave
    lp = adj_wave
    return lp
