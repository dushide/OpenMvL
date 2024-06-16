import copy
import os
import warnings
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from args_tNet import parameter_parser
from util.loadMatData import load_data, features_to_Lap, features_to_Adj, sparse_mx_to_torch_sparse_tensor, load_data_2
import scipy.sparse as sp
from util.label_utils import reassign_labels, special_train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from TrustworthyNet import TrustworthyNet_classfier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from datasets import construct_H_with_KNN
import util.hypergraph_utils as hgut
from util.config import load_config
np.set_printoptions(threshold=np.inf)
from scipy import sparse
import scipy.sparse as ss
import scipy.io as sio
def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    F1_macro = metrics.f1_score(labels_true, labels_pred, average='macro')
    F1_micro = metrics.f1_score(labels_true, labels_pred, average='micro')

    return ACC, F1_macro, F1_micro

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

def evaluate(logits, mask_indices, show_matrix=False, filter_unseen=True,threshold=None):

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
        probs = probs[mask_indices]
        probs = gather_nd(probs, torch.stack([torch.arange(masked_logits.size(0)), masked_y_pred], axis=1))
        probs = probs.cpu().detach().numpy()
        masked_y_pred = masked_y_pred.numpy()
        print("mean: ", probs.mean())
        if threshold is None:
            threshold = (probs[masked_y_true != args.unseen_label_index].mean()+probs[masked_y_true == args.unseen_label_index].mean())/2.0
            print("auto meanS: ", threshold)
        # threshold
        masked_y_pred[probs < threshold] = args.unseen_label_index

    else:
        masked_y_pred = masked_y_pred.numpy()

    masked_y_true = masked_y_true
    accuracy = accuracy_score(masked_y_true, masked_y_pred)
    macro_f_score = f1_score(masked_y_true, masked_y_pred, average="macro")

    if show_matrix:
        print(classification_report(masked_y_true, masked_y_pred))
        print(confusion_matrix(masked_y_true, masked_y_pred))

    if filter_unseen:
        return accuracy, macro_f_score, threshold
    else:
        return accuracy, macro_f_score

def logits_to_probs(logits):
    if args.use_softmax:
        probs = torch.softmax(logits, dim=1)
        # probs = F.log_softmax(logits, dim=1)
    else:
        probs = torch.nn.sigmoid(logits)
    return probs

def mse_loss(p, alpha, c):
    S = torch.sum(alpha, dim=1, keepdim=True) # Dirichlet strength
    E = alpha - 1 # Evidence
    m = alpha / S # Confidence
    # label = F.one_hot(p, num_classes=c)
    A = torch.sum((p - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - p) + 1
    C = KL(alp, c)
    return (A + B) + C

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

def compute_loss_0(outputs, labels, mask_indices):

    logits = outputs
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
    loss_unseen = (unmasked_probs * torch.log(unmasked_probs)).mean()

    #####################################################################################

    logits = F.log_softmax(logits, dim=1)
    masked_logits = logits[mask_indices]
    masked_y_true = labels[mask_indices].to(device)
    # h = F.one_hot(masked_y_true, num_classes)
    loss_seen = torch.nn.NLLLoss()(masked_logits, masked_y_true)
    #####################################################################################

    loss = args.alpha_1 * loss_seen + args.beta_1 * loss_unseen



    return loss

def compute_loss(outputs, alpha, labels, mask_indices):

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
    loss_unseen = ce_loss(unmasked_preds[topk_indices], unmasked_probs, num_classes).mean()

    #################################################################################

    masked_logits = logits[mask_indices]
    masked_y_true = labels[mask_indices].to(device)
    loss_seen = ce_loss(masked_y_true, masked_logits, num_classes).mean()
    #####################################################################################

    loss = args.alpha * loss_seen + args.beta * loss_unseen

    return loss
def compute_loss_1(outputs, alpha, labels, mask_indices):

    logits = alpha
    labels = labels.long()

    ##################################################

    all_indices = np.arange(0, logits.size(0))
    unmasked_indices = np.delete(all_indices, mask_indices)
    unmasked_logits = logits[unmasked_indices]
    unmasked_preds = torch.argmax(unmasked_logits, 1).to(device)
    loss_unseen = ce_loss_1(unmasked_preds, unmasked_logits, num_classes).mean()

    #####################################################################################

    masked_logits = logits[mask_indices]
    masked_y_true = labels[mask_indices].to(device)
    loss_seen = ce_loss(masked_y_true, masked_logits, num_classes).mean()
    #####################################################################################

    loss = args.alpha * loss_seen + args.beta * loss_unseen

    return loss

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

    loss = args.alpha * loss_seen + args.beta * loss_unseen

    return loss
#########################################################################################################################################################################


#########################################################################################################################################################################
#########################################################################################################################################################################

def train(args, device, features, labels):
    print(
        "use_hypergraph:{}, unseen_num:{},fusion:{}, active:{}, block:{}, training_rate:{}, alpha:{}, beta:{}, gamma:{}, epoch:{} \n".format(
            args.use_hypergraph, args.unseen_num, args.fusion_type, args.active, args.block, args.training_rate,
            args.alpha, args.beta, args.gamma, args.epoch))
    model = TrustworthyNet_classfier(n_feats, n_view, num_classes, n, args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ACC = 0
    begin_time = time.time()
    for epoch in range(args.epoch):
        t = time.time()
        E_list, Z_logit, A_list, alpha = model(features, lap, args.active)
        optimizer.zero_grad()
        loss = 0
        for i in range(n_view):
            loss += compute_loss_2(Z_logit, A_list[i], y_true, train_indices)
        loss += compute_loss_0(Z_logit, y_true, train_indices)
        loss += compute_loss_2(Z_logit, alpha, y_true, train_indices)
        try:
            loss.backward()
        except RuntimeError:
            print("element 0 of tensors does not require grad and does not have a grad_fn\n")
            break;
        else:
            optimizer.step()
            # scheduler.step(loss)
            train_accuracy, _ = evaluate(Z_logit.cpu(), train_indices, filter_unseen=False)
            valid_accuracy, valid_macro_f_score, threshold = evaluate(Z_logit.cpu(), valid_indices, filter_unseen=True)
            if valid_accuracy >= best_ACC:
                    best_ACC = valid_accuracy

                    best_model_wts = copy.deepcopy(model.state_dict())

        finally:

            print("Epoch:", '%04d' % (epoch + 1), "best_acc=", "{:.5f}".format(best_ACC),
                  "train_loss=", "{:.5f}".format(loss.item()),
                  "train_acc=", "{:.5f}".format(train_accuracy), "val_acc=", "{:.5f}".format(valid_accuracy),
                  "threshold=", "{:.5f}".format(threshold), "time=", "{:.5f}".format(time.time() - t))
    run_time = time.time() - begin_time
    model.load_state_dict(best_model_wts)
    with torch.no_grad():
        E_list, Z_logit, A_list, alpha = model(features, lap, args.active)

        test_accuracy, test_macro_f_score, _ = evaluate(Z_logit.cpu(), test_indices, show_matrix=False,
                                                        filter_unseen=True,
                                                        threshold=threshold)

    if args.save_results:
        with open(args.save_path, "a") as f:
            with open(args.save_path, "a") as f:
                f.write(
                    "use_hypergraph:{}, unseen_num:{},fusion:{}, active:{}, block:{}, training_rate:{}, alpha:{}, beta:{}, gamma:{}, epoch:{} \n".format(
                        args.use_hypergraph, args.unseen_num, args.fusion_type, args.active, args.block,
                        args.training_rate, args.alpha, args.beta, args.gamma, args.epoch))
                f.write("{}:{}\n".format(dataset, dict(
                    zip(['acc', 'F1_macro', 'time'],
                        [round(test_accuracy * 100, 2), round(test_macro_f_score * 100, 2), run_time]))))

def construct_hypergraph(feature_list, knn,device):
    lap = []
    for i in range(n_view):
        direction_judge = '../hyperlap_matrix/' + dataset + '/' + 'v' + str(i) + '_knn' + str(knn) + '_hyperlap.npz'
        if os.path.exists(direction_judge):
            print("Loading the hyperlap matrix of " + str(i) + "th view of " + dataset)
            temp_lap = ss.load_npz(direction_judge)
            lap.append(torch.from_numpy(temp_lap.todense()).float().to(device))

        else:

            print("Constructing the hyperlap matrix of " + str(i) + "th view of " + dataset)

            H = construct_H_with_KNN([feature_list[i]], knn, split_diff_scale=True)
            G = hgut.generate_G_from_H(H)
            temp_lap = np.identity(len(G)) - G
            save_direction = '../hyperlap_matrix/' + dataset + '/'
            if not os.path.exists(save_direction):
                os.makedirs(save_direction)

            print("Saving the adjacency matrix to " + save_direction)
            ss.save_npz(save_direction + 'v' + str(i) + '_knn' + str(knn) + '_hyperlap.npz', sparse.csr_matrix(temp_lap[0]))

            lap.append(torch.from_numpy(temp_lap[0]).to(torch.float32).to(device))
    return lap


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
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    dataset_dict = {1:  'Caltech101all',  2: 'Hdigit',3: 'MITIndoor', 4: 'MNIST10k',
               5: 'NoisyMNIST_30000', 6: 'NUSWIDE',
               7: 'scene15', 8: 'Youtube'}

    select_dataset = [2]


    for ii in select_dataset:
        dataset=dataset_dict[ii]
        if args.use_hypergraph:
            config = load_config('./config/unseen'+str(args.unseen_num)+'_'+args.active+'_hyper.yaml')
            config2 = load_config('./config/gamma_unseen' + str(args.unseen_num) + '_' + args.active + '_hyper.yaml')
        else:
            config = load_config('./config/unseen' + str(args.unseen_num) + '_' + args.active + '.yaml')
            config2= load_config('./config/gamma_unseen' + str(args.unseen_num) + '_' + args.active + '.yaml')

        args.block=config[dataset]
        args.gamma=config2[dataset]
        if args.fix_seed:
            torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        print("========================", dataset)
        features, labels= load_data_2(dataset, args.data_path)
        labels = labels + 1


        n_view = len(features)
        n_feats = [x.shape[1] for x in features]
        n = features[0].shape[0]
        n_classes = len(np.unique(labels))

        feature_list = []
        for i in range(n_view):
            feature_list.append(features[i])
            features[i] = torch.from_numpy(features[i] / 1.0).float().to(device)

        if args.unseen_num == 1:
            original_num_classes = np.max(labels) + 1
        elif args.unseen_num != 1:
            original_num_classes = np.max(labels) + 1

        seen_labels = list(range(1, original_num_classes - args.unseen_num))
        y_true = reassign_labels(labels, seen_labels, args.unseen_label_index)
        for i in range(len(y_true)):
            if y_true[i]!=-100:
                y_true[i] = y_true[i]+1


        train_indices, test_valid_indices = special_train_test_split(y_true, args.unseen_label_index,
                                                                     test_size=1 - args.training_rate)
        test_indices, valid_indices = train_test_split(test_valid_indices, test_size=args.valid_rate / (1 - args.training_rate))


        if args.unseen_num == 1:
            num_classes = np.max(y_true) + 1
        elif args.unseen_num != 1:
            num_classes = np.max(y_true) + 1

        y_true = torch.from_numpy(y_true)
        print('data:{}\tseen_labels:{}\tuse_softmax:{}\trandom_seed:{}\tunseen_num:{}\tnum_classes:{}'.format(
            dataset,
            seen_labels,
            args.use_softmax,
            args.seed,
            args.unseen_num,
            num_classes))

        print(dataset, n, n_view, n_feats,n_classes)
        labels = torch.from_numpy(labels).long().to(device)

        knn = int(n/n_classes)

        if args.use_hypergraph:
            lap=construct_hypergraph(feature_list, knn, device)
        else:
            true_lap, lap = features_to_Lap(dataset,feature_list,device, knn)
        try:
            if args.fix_seed:
                torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
                torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
            train(args, device, features, y_true)
        except Exception as e:
            with open(args.save_path, "a") as f:
                f.write("-----------------")
            print(e)
        finally:
            with open(args.save_path, "a") as f:
                f.write("\n")
                continue

