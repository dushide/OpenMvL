import os

import torch

import datasets
import numpy as np
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale

def dataloader(dataset_dir, dataset_name, random_ratio):
    '''
    先shuffle，读取数据，生成adj，然后再生成排列来确定标签。
    '''
    data_dir = dataset_dir # /home/zhongly/coding/MultiView/Multi_Dataset

    print("Generating training data.")
    features, gnd = datasets.loadMatData(dataset_dir, dataset_name) # D://CL3/Multi_View
    # print(len(features), type(features), features[0][0].shape)
    features = feature_normalization(features, dataset_name)

    # save data
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if dataset_name == "Reuters" or dataset_name == "Reuters_mini":
        for idx, feature in enumerate(features[0]):
            features[0][idx] = feature.todense()
    p_labeled, p_unlabeled = generate_permutation(gnd, random_ratio)

    return features, gnd, p_labeled, p_unlabeled

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def feature_normalization(features, dataset_name, normalization_type = 'normalize'):
    if dataset_name == "Pubmed":
        for idx, fea in enumerate(features):
            if normalization_type == 'normalize':
                features[idx] = normalize(fea)
    else:
        for idx, fea in enumerate(features[0]):
            if normalization_type == 'minmax_scale':
                features[0][idx] = minmax_scale(fea)
            elif normalization_type == 'maxabs_scale':
                features[0][idx] = maxabs_scale(fea)
            elif normalization_type == 'normalize':
                features[0][idx] = normalize(fea)
            elif normalization_type == 'robust_scale':
                features[0][idx] = robust_scale(fea)
            elif normalization_type == 'scale':
                features[0][idx] = scale(fea)
            elif normalization_type == '255':
                features[0][idx] = np.divide(fea, 255.)
            elif normalization_type == '50':
                features[0][idx] = np.divide(fea, 50.)
            else:
                print("Please enter a correct normalization type!")
            # pdb.set_trace()
    return features


def count_each_class_num(gnd):
    '''
    Count the number of samples in each class
    '''
    count_dict = {}
    for label in gnd:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict

def data_shuffle(features, gnd):
    N = len(gnd)

    # shuffle
    perm = np.random.permutation(N)
    gnd = gnd[perm]
    for idx, fea in enumerate(features[0]):
        features[0][idx] = fea[perm]
    return features, gnd

def generate_permutation(gnd, ratio):
    '''
    Generate permutation for training (labeled) and testing (unlabeled) data.
    '''
    N = len(gnd)

    each_class_num = count_each_class_num(gnd)
    labeled_each_class_num = {} ## number of labeled samples for each class
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1) # min is 1

    # Data shuffle
    data_idx = np.random.permutation(range(N))

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    for idx in data_idx:
        label = gnd[idx]
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(idx)
        else:
            p_unlabeled.append(idx)
    return p_labeled, p_unlabeled

#########################################################################

def load_ft(dataset,on_dataset,ratio):
    features, gnd, p_labeled, p_unlabeled = dataloader(dataset,on_dataset,ratio)
    lbls = gnd
    if lbls.min() == 1:
        lbls = lbls - 1
    fts = features
    idx_train = p_labeled
    idx_test = p_unlabeled
    return fts, lbls, idx_train, idx_test
