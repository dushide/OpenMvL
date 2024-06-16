import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
import sys



class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=64):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb


class TrustworthyNet_classfier(nn.Module):
    def __init__(self, nfeats, n_view, n_classes, n, args, device):
        """
        build TrustworthyNet for classfier.
        :param nfeats: list of feature dimensions for each view
        :param n_view: number of views
        :param n_classes: number of clusters
        :param n: number of samples
        :param args: Relevant parameters required to build the network
        """
        super(TrustworthyNet_classfier, self).__init__()
        self.n_classes = n_classes
        #  number of differentiable blocks
        self.block = args.block
        # the initial value of the threshold
        self.theta = nn.Parameter(torch.FloatTensor([args.thre]), requires_grad=True).to(device)
        self.n_view = n_view
        self.n = n
        self.ZZ_init = []
        self.fusion_type = args.fusion_type
        self.bn_input_01 = nn.BatchNorm1d(self.n_classes, momentum=0.5).to(device)
        self.device = device
        if self.fusion_type != 'trust':
            self.fusionlayer = FusionLayer(n_view, self.fusion_type,self.n_classes, hidden_size=64)

        for i in range(n_view):
            exec('self.block{} = Block(n_classes, {}, args.gamma, device)'.format(i, nfeats[i]))

        for ij in range(n_view):
            self.ZZ_init.append(nn.Linear(nfeats[ij], n_classes).to(device))


    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.n_classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.n_classes, 1), b[1].view(-1, 1, self.n_classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.n_classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def self_active_l1(self, u):
        return F.selu(u - self.theta) - F.selu(-1.0 * u - self.theta)

    def self_active_l21(self, x):
        nw = torch.norm(x)
        if nw > self.theta:
            x = (nw - 1 / self.theta) * x / nw
        else:
            x = torch.zeros_like(x)
        return (x)

    def forward(self, features, lap, active):
        output_z = []
        output_z_1 = []
        E_list = []
        A_list = []
        out_tmp = 0
        if active == 'l1':
            for j in range(0, self.n_view):
                out_tmp = self.ZZ_init[j](features[j] / 1.0)
                output_z.append(self.self_active_l1(out_tmp).to(self.device))
                output_z_1.append((self.self_active_l1(out_tmp) + 1).to(self.device))
            E_list.append(output_z)
            A_list.append(output_z_1)
        elif active == "l21":
            for j in range(0, self.n_view):
                out_tmp = self.ZZ_init[j](features[j] / 1.0)
                output_z.append(self.self_active_l21(out_tmp).to(self.device))
                output_z_1.append((self.self_active_l21(out_tmp) + 1).to(self.device))
            E_list.append(output_z)
            A_list.append(output_z_1)

        for i in range(0, self.block):
            for j in range(0, self.n_view):
                exec('h{} = self.block{}(E_list[-1][j], lap[j], features[j] / 1.0)'.format(j, j, j, j, j))  # evi_v
                if active == 'l1':
                    exec('output_z.append(self.self_active_l1((h{})))'.format(j))  # # evi_v
                    exec('output_z_1.append(self.self_active_l1((h{})) + 1)'.format(j))  # # a_v

                elif active == "l21":
                    exec('output_z.append(self.self_active_l21(h{}))'.format(j))  # # evi_v
                    exec('output_z_1.append(self.self_active_l21(h{}) + 1)'.format(j))  # # a_v
            exec('E_list.append(output_z)'.format(j))  # # evi_v
            exec('A_list.append(output_z_1)'.format(j))  # a_v

            if self.fusion_type == 'trust':
                alpha = self.DS_Combin(A_list[-1])  # a
                evi = alpha - 1  # evi
            else:
                evi = self.fusionlayer(E_list[-1])

        return E_list[-1], evi, A_list[-1], alpha


class Block(Module):
    # differentiable network block
    def __init__(self, out_features, nfea, gamma, device):
        super(Block, self).__init__()
        self.S_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.S = nn.Linear(out_features, out_features).to(device)

        self.G_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.G = nn.Linear(out_features, out_features).to(device)

        self.U_norm = nn.BatchNorm1d(nfea, momentum=0.6).to(device)
        self.U = nn.Linear(nfea, out_features).to(device)
        self.gamma=gamma
        self.device = device


    def forward(self, input, lap, view):
        input1 = self.S((input))
        input2 = self.U((view))
        lap = lap * lap
        output = torch.mm(lap, input)
        output = self.G((output))
        output = input1 + input2 - self.gamma * output
        return output