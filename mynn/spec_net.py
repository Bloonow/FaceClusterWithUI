from math import sqrt

import torch
from torch import nn

from mynn.layer import stack_layers


class SpectralNet(nn.Module):
    def __init__(self, in_dim, arch, view_size, device_str='cpu'):
        """
        :param in_dim: 输入数据的维数
        :param arch: 表示网络的结构，一个列表，每个元素结构为：{'layer': 'linear', 'size': 1024, 'activation': 'relu'}
        """
        super().__init__()
        self.view_size = view_size

        layers_mv_list = []
        for vi in range(view_size):
            layers = stack_layers(in_dim, arch)
            layers.append(OrthogonalLayer())
            layers_mv_list.append(layers)

        self.layers_mv_list = nn.ModuleList(layers_mv_list)
        self.to(device_str)

    def forward(self, data_batch_mv_list):
        """
        :param data_batch_mv_list: 一批多视图数据，结构为：
        [ tensor([<view_0 data>, <>, ..., <>]), tensor([<view_1 data>, <>, ..., <>]), ... ]
        """
        out_batch_mv_list = []
        for vi in range(self.view_size):
            out_batch = self.layers_mv_list[vi](data_batch_mv_list[vi])
            out_batch_mv_list.append(out_batch)
        return out_batch_mv_list


class OrthogonalLayer(nn.Module):
    def __init__(self, epsilon=5e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, X):
        n, dim = X.size()
        device = X.device

        # X_npy = X.clone().detach().cpu()
        # X_2_npy = np.matmul(np.transpose(X_npy), X_npy) + np.eye(dim) * self.epsilon
        # L_npy = np.linalg.cholesky(X_2_npy)
        # Q_npy = np.transpose(np.linalg.inv(L_npy)) * sqrt(n)
        # Q_npy = Q_npy.astype(np.float32)
        # Q = torch.from_numpy(Q_npy).to(device)

        X_2 = torch.matmul(X.t(), X) + torch.eye(dim, device=device) * self.epsilon
        L = torch.linalg.cholesky(X_2)
        Q = L.inverse().t() * sqrt(n)

        return torch.matmul(X, Q)


def get_new_spec_net(in_dim, arch, view_size):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    arch[-1]['activation'] = 'tanh'
    return SpectralNet(in_dim, arch, view_size=view_size, device_str=device_str)


def spec_loss_fn_one(data_batch_mv_list, W_mv_list):
    """
    计算该批数据所有视图的损失值 L1
    :param data_batch_mv_list: 所有视图的SpectralNet输出tensor列表
    :param W_mv_list: 所有视图的SiameseNet生成的affinity近似矩阵
    """
    view_size = len(data_batch_mv_list)
    n, d = data_batch_mv_list[0].size()
    device = data_batch_mv_list[0].device

    loss = torch.zeros((1,), device=device)
    for vi in range(view_size):
        data_batch = data_batch_mv_list[vi]
        W = W_mv_list[vi]

        # 空间换时间
        A = torch.empty((n, n, d), device=device)
        B = torch.empty((n, n, d), device=device)
        for i in range(n):
            A[i, :] = data_batch[i]
            B[:, i] = data_batch[i]
        Dy = torch.sum(torch.square(A - B), dim=2)

        # loss_in_view = torch.sum(W * Dy) / ( n * n )
        loss_in_view = torch.mean(W * Dy)
        loss += loss_in_view

    loss /= view_size
    return loss


def spec_loss_fn_two(data_batch_mv_list):
    """
    计算该批数据所有视图的损失值 L2
    :param data_batch_mv_list: 所有视图的SpectralNet输出tensor列表
    """
    view_size = len(data_batch_mv_list)
    device = data_batch_mv_list[0].device

    loss = torch.zeros((1,), device=device)
    for i in range(view_size):
        for j in range(i + 1, view_size):
            view_i_batch = data_batch_mv_list[i]
            view_j_batch = data_batch_mv_list[j]
            dist_square_batch = torch.sum(torch.square(view_i_batch - view_j_batch), dim=1)
            # loss += torch.sum(dist_square_batch) / batch_size
            loss += torch.mean(dist_square_batch)

    loss /= (view_size * view_size)
    return loss


def spec_loss_fn(data_batch_mv_list, W_mv_list, lamb=0.001):
    loss_1 = spec_loss_fn_one(data_batch_mv_list, W_mv_list)
    loss_2 = spec_loss_fn_two(data_batch_mv_list)
    return (1 - lamb) * loss_1 + lamb * loss_2
