import numpy as np

def load_feature_construct_H(feature,
                             K_neigs=[20],
                             is_probH=True,
                             split_diff_scale=False):
    """

    :param data_dir: directory of feature data
    :param m_prob: parameter in hypergraph incidence matrix construction
    :param K_neigs: the number of neighbor expansion
    :param is_probH: probability Vertex-Edge matrix or binary
    :param use_mvcnn_feature:
    :param use_gvcnn_feature:
    :param use_mvcnn_feature_for_structure:
    :param use_gvcnn_feature_for_structure:
    :return:
    """

    # construct feature matrix

    # construct hypergraph incidence matrix
    print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    H = construct_H_with_KNN( feature)

    return H

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    从超图节点距离矩阵中构建超图发生率矩阵
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0] #3327
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    low_H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        farthest_idx = np.array(np.argsort(dis_vec, axis=-1)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                if node_idx == center_idx:
                    H[node_idx, center_idx] = 2*np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
                else:
                    H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0

        for node_idx in farthest_idx[:k_neig]:
            if is_probH:
                low_H[node_idx, center_idx] = -np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = -1.0

    return 0.7*H+0.3* low_H

def feature_concat(F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    串联多种模式的特征。如果一个特征矩阵的维度超过两个，该函数将把它减少为两个维度（使用最后一个维度作为特征维度，其他维度将被融合为对象维度）
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features

def construct_H_with_KNN(X, K_neigs=[20], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    从原始节点特征矩阵启动多尺度超图顶点-边缘矩阵
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion 邻居扩张的数量
    :param split_diff_scale: whether split hyperedge group at different neighbor scale 是否在不同的邻居范围内分割超群组
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    H = []
    if(isinstance(X,list)):
        for i in range (len(X)):

            if type(K_neigs) == int:
                K_neigs = [K_neigs]

            dis_mat = Eu_dis(X[i])
            for k_neig in K_neigs:
                H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
                if not split_diff_scale:
                    H = hyperedge_concat(H, H_tmp)
                else:
                    H.append(H_tmp)

    return H

def construct_H_with_KNN2(X, K_neigs=[20], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    从原始节点特征矩阵启动多尺度超图顶点-边缘矩阵
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion 邻居扩张的数量
    :param split_diff_scale: whether split hyperedge group at different neighbor scale 是否在不同的邻居范围内分割超群组
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    H = []
    for i in range (len(X)):
        if type(K_neigs) == int:
            K_neigs = [K_neigs]
        dis_mat = Eu_dis(X[i])
        for k_neig in K_neigs:
            H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
            H.append(H_tmp)
    return H

def construct_H_with_adj(X, dis_mat,  K_neigs=[20],split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    从原始节点特征矩阵启动多尺度超图顶点-边缘矩阵
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion 邻居扩张的数量
    :param split_diff_scale: whether split hyperedge group at different neighbor scale 是否在不同的邻居范围内分割超群组
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    H = []
    for i in range (len(X)):
        if type(K_neigs) == int:
            K_neigs = [K_neigs]
        # dis_mat = Eu_dis(X[i])
        for k_neig in K_neigs:
            H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
            H.append(H_tmp)
    return H
def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H