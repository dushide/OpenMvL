import copy
import os
import sys
from argparse import ArgumentParser
import warnings
import time
import random
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from util.loadMatData import load_data, features_to_Lap, sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp
from util.label_utils import reassign_labels, special_train_test_split
from sklearn.model_selection import train_test_split
from TrustworthyNet import TrustworthyNet_classfier
from sklearn.metrics import accuracy_score, f1_score
from util.ECE_metrics import get_metrics
from util.config import load_config

np.set_printoptions(threshold=np.inf)



def gather_nd(params, indices):

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    out = torch.take(params, idx)

    return out.view(out_shape)
#
def evaluate(logits, mask_indices, uncertainty, filter_unseen=True, threshold=None):
    if isinstance(logits, list):
        logits = logits.mean(0)
        if args.use_softmax:
            probs_list = torch.softmax(logits, axis=1)
        else:
            probs_list = torch.sigmoid(logits)
        probs = probs_list.mean(0)
    else:
        probs = logits_to_probs(logits)
    masked_logits = logits[mask_indices]
    masked_y_pred = torch.argmax(masked_logits, 1)
    masked_y_true = y_true[mask_indices]
    if filter_unseen:
        uncertainty = uncertainty[mask_indices]
        # Gather uncertainties for the predicted classes
        probs= gather_nd(probs, torch.stack([torch.arange(masked_logits.size(0)), masked_y_pred], axis=1))
        probs= probs.cpu().detach().numpy()
        uncertainty = uncertainty.cpu().detach().numpy().flatten()
        masked_y_pred = masked_y_pred.numpy()
        print("mean uncertainty: ", uncertainty.mean())
        # Apply threshold to classify samples as unknown if their uncertainty is above the threshold
        masked_y_pred[uncertainty >= 0.7] = args.unseen_label_index
    else:
        masked_y_pred = masked_y_pred.numpy()
    accuracy = accuracy_score(masked_y_true, masked_y_pred)
    macro_f_score = f1_score(masked_y_true, masked_y_pred, average="macro")
    return accuracy, macro_f_score, probs

def logits_to_probs(logits):
    if args.use_softmax:
        probs = torch.softmax(logits, dim=1)
    else:
        probs = torch.nn.sigmoid(logits)
    return probs

def compute_loss(outputs, labels, mask_indices):

    logits = outputs
    labels = labels.long()
    all_indices = np.arange(0, logits.size(0))
    unmasked_indices = np.delete(all_indices, mask_indices)
    unmasked_logits = logits[unmasked_indices]

    unmasked_probs = logits_to_probs(unmasked_logits)
    unmasked_probs = torch.clamp(unmasked_probs, 1e-7, 1.0)
    unmasked_preds = torch.argmax(unmasked_probs, 1).to(device)
    unmasked_prob = gather_nd(unmasked_probs, torch.stack([torch.arange(unmasked_logits.size(0)).to(device), unmasked_preds], axis=1))

    topk_indices_a = torch.logical_and(torch.greater(unmasked_prob, 1.0 / num_classes), torch.less(unmasked_prob, 0.5))
    topk_indices_b = torch.arange(topk_indices_a.size(0)).to(device)
    topk_indices = torch.masked_select(topk_indices_b, topk_indices_a)

    unmasked_probs = unmasked_probs[topk_indices]
    loss_unseen = (unmasked_probs * torch.log(unmasked_probs)).mean()

    #####################################################################################

    logits = F.log_softmax(logits, dim=1)
    masked_logits = logits[mask_indices]
    masked_y_true = labels[mask_indices].to(device)
    loss_seen = torch.nn.NLLLoss()(masked_logits, masked_y_true)
    #####################################################################################

    return loss_seen + loss_unseen

#########################################################################################################################################################################

def ce_loss(p, alpha, c):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    #annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1

    B = KL(alp, c)

    return (A + B)

def ce_loss_1(p, alpha, c):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(alpha) - torch.digamma(S)), dim=1, keepdim=True)

    alp = E * (1 - label) + 1
    B = KL(alp, c)

    return (A + B)
def KL(alpha, c):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def compute_loss_2(outputs, alpha, labels, mask_indices):

    logits = alpha
    labels = labels.long()

    ##################################################

    all_indices = np.arange(0, logits.size(0))
    unmasked_indices = np.delete(all_indices, mask_indices)
    unmasked_logits = logits[unmasked_indices]

    unmasked_probs = logits_to_probs(unmasked_logits)
    unmasked_probs = torch.clamp(unmasked_probs, 1e-7, 1.0)
    unmasked_preds = torch.argmax(unmasked_probs, 1).to(device)
    unmasked_prob = gather_nd(unmasked_probs, torch.stack([torch.arange(unmasked_logits.size(0)).to(device), unmasked_preds], axis=1))

    topk_indices_a = torch.logical_and(torch.greater(unmasked_prob, 1.0 / num_classes), torch.less(unmasked_prob, 0.5))
    topk_indices_b = torch.arange(topk_indices_a.size(0)).to(device)
    topk_indices = torch.masked_select(topk_indices_b, topk_indices_a)

    unmasked_probs = unmasked_probs[topk_indices]
    loss_unseen = ce_loss_1(unmasked_preds[topk_indices], unmasked_probs, num_classes).mean()

    #####################################################################################

    masked_logits = logits[mask_indices]
    masked_y_true = labels[mask_indices].to(device)
    loss_seen = ce_loss(masked_y_true, masked_logits, num_classes).mean()
    #####################################################################################

    return (loss_seen+loss_unseen)/len(train_indices)
def train(args, device, features, labels):


    model = TrustworthyNet_classfier(n_feats, n_view, num_classes, n, args,device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ACC = 0
    loss_list = []
    acc_list = []
    f1_macro_list = []
    begin_time = time.time()
    for epoch in range(args.epoch):
        t = time.time()
        Z_logit, A_list, alpha, u = model(features, lap, args.active)
        optimizer.zero_grad()
        # loss =compute_loss(Z_logit, y_true, train_indices)
        loss =args.hp* compute_loss(Z_logit, y_true, train_indices)

        loss += args.lambda1*compute_loss_2(Z_logit, alpha, y_true, train_indices)
        for i in range(n_view):
            loss += args.lambda2*compute_loss_2(Z_logit, A_list[i], y_true, train_indices)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        with torch.no_grad():
            Z_logit, A_list, alpha, u = model(features, lap, args.active)
            valid_accuracy, valid_macro_f_score, _= evaluate(Z_logit.cpu(), valid_indices,u, filter_unseen=True)
            if valid_accuracy >= best_ACC:
                best_ACC = valid_accuracy

                best_model_wts = copy.deepcopy(model.state_dict())

        print("Epoch:", '%04d' % (epoch + 1), "best_acc=", "{:.5f}".format(best_ACC),
                  "train_loss=", "{:.5f}".format(loss),"val_acc=", "{:.5f}".format(valid_accuracy), "time=", "{:.5f}".format(time.time() - t))

    model.load_state_dict(best_model_wts)
    with torch.no_grad():
        Z_logit, A_list, alpha,u = model(features, lap, args.active)
        test_accuracy, test_macro_f_score, probs = evaluate(Z_logit.cpu(), test_indices, u )
        binary_label = torch.where(y_true[test_indices] == args.unseen_label_index, 0, 1).detach().numpy()
        ECE, MCE = get_metrics(binary_label, probs)


    run_time=time.time()-begin_time
    if args.save_results:
        with open(args.file, "a") as f:
            f.write(f"unseen_num:{args.unseen_num}, blocks:{args.block}, gamma:{args.alpha}.lambda1:{args.lambda1}, lambda2:{args.lambda2},knn:{knn}\n")
            f.write("{}:{}\n".format(data, dict(
                zip(['acc', 'F1_macro', 'ECE', 'MCE','time'], [round(test_accuracy * 100, 2),
                                                               round(test_macro_f_score * 100, 2),
                                                               round(ECE * 100, 2),
                                                               round(MCE * 100, 2), run_time]))))

## Parameter setting
def parameter_parser():
    parser = ArgumentParser()

    current_dir = sys.path[0]
    parser.add_argument("--path", type=str, default=current_dir)

    parser.add_argument("--data_path", type=str, default="/data/", help="Path of datasets.")
    parser.add_argument("--file", type=str, default="./result.txt", help="Path of datasets.")

    # input_type； choose features or similarity graphs to learn a multi-variate heterogeneous representation.
    parser.add_argument("--save_results", action='store_true', default=True, help="Save experimental result.")
    parser.add_argument("--use_softmax", action='store_true', default=True)
    parser.add_argument("--resp", type=int, default=1)
    parser.add_argument("--unseen_label_index", type=int, default=-100)
    parser.add_argument("--fusion_type", type=str, default="trust",
                        help="Fusion Methods: trust average weight attention")

    parser.add_argument("--active", type=str, default="l1", help="l21 or l1")
    # the type of regularizer with Prox_h()

    parser.add_argument("--device", default="0", type=str, required=False)
    parser.add_argument("--fix_seed", action='store_true', default=True, help="")
    parser.add_argument("--seed", type=int, default=40, help="Random seed, default is 42.")
    parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument("--training_rate", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--valid_rate", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--weight_decay", type=float, default=0.15, help="Weight decay")
    parser.add_argument("--in_size", type=int, default=102)

    parser.add_argument('--epoch', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--unseen_num', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--block', type=int, default=1,
                        help='block')  # for the example dataset, block can set 2 and more than 2
    parser.add_argument('--thre', type=float, default=0.1)
    return parser.parse_args()


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parameter_parser()
    if args.fix_seed:
        seed = 20
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    args.device = '0'

    args.lambda1 = 1
    args.lambda2 =1

    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)


    dataset_dict = {1:  'Caltech101all',  2: 'Hdigit',3: 'MITIndoor', 4: 'MNIST10k',
               5: 'NoisyMNIST_30000', 6: "NUSWide20k",
               7: 'scene15', 8: 'Youtube'}
    select_dataset = [8]
    args.hp = 10000
    for ii in select_dataset:
        data = dataset_dict[ii]
        print("========================", data)
        features, labels= load_data(dataset_dict[ii], './data/')
        config = load_config(f'./config/{data}.yaml')
        args.alpha=config['alpha']
        n_view = len(features)
        n_feats = [x.shape[1] for x in features]
        n = features[0].shape[0]
        n_classes = len(np.unique(labels))


        # open2 = (1 - openness) * (1 - openness)
        # args.unseen_num = round((1 - open2 / (2 - open2)) * n_classes)
        print("unseen_num:%d" % args.unseen_num)


        feature_list = []
        for i in range(n_view):
            feature_list.append(features[i])
            features[i] = torch.from_numpy(features[i] / 1.0).float().to(device)

        original_num_classes = np.max(labels) + 1
        seen_labels = list(range(original_num_classes -args.unseen_num))
        y_true = reassign_labels(labels, seen_labels, args.unseen_label_index)

        train_indices, test_valid_indices = special_train_test_split(y_true, args.unseen_label_index,
                                                                     test_size=1 - args.training_rate)
        test_indices, valid_indices = train_test_split(test_valid_indices, test_size=args.valid_rate / (1 - args.training_rate))


        num_classes = np.max(y_true) + 1
        y_true = torch.from_numpy(y_true)
        print('data:{}\tseen_labels:{}\tuse_softmax:{}\t\tunseen_num:{}\tnum_classes:{}'.format(
            dataset_dict[ii],
            seen_labels,
            args.use_softmax,
            args.unseen_num,
            num_classes))

        print(data, n, n_view, n_feats,n_classes)
        labels = torch.from_numpy(labels).long().to(device)

        knn = int(n/n_classes*0.5)
        lap = features_to_Lap(feature_list, knn)

        for i in range(n_view):
            lap[i] = lap[i].to_dense().to(device)

        train(args, device, features, labels)

