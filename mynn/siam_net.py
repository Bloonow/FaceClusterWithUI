import torch
from torch import nn

from mynn.layer import stack_layers
from utils import C


class SiameseNet(nn.Module):
    def __init__(self, in_dim, arch, device_str='cpu'):
        """
        :param in_dim: 输入数据的维数
        :param arch: 表示网络的结构，一个列表，每个元素结构为：{'layer': 'linear', 'size': 1024, 'activation': 'relu'}
        """
        super().__init__()
        self.layers = stack_layers(in_dim, arch)
        self.to(device_str)

    def forward(self, data_batch):
        """
        :param data_batch: 一批数据，结构为：tensor([<data point>, ..., <data point>])
        """
        return self.layers(data_batch)


def get_new_siam_net(in_dim, arch):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SiameseNet(in_dim, arch, device_str=device_str)


def euclidean_distance(a_batch, b_batch):
    """
    求两批向量对应两两之间的欧氏距离
    """
    n, d = a_batch.size()
    device = a_batch.device

    epsilon_batch = torch.ones((n,), device=device) * C.epsilon()

    dist_square_batch = torch.maximum(
        torch.sum(torch.square(a_batch - b_batch), dim=1),
        epsilon_batch
    )
    return torch.sqrt(dist_square_batch)


def contrastive_loss(true_batch, pred_batch, pos_mask=0.05, neg_mask=1):
    """
    Contrastive loss from Hadsell-et-al. http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    n = true_batch.size()[0]
    device = true_batch.device

    pos_mask_batch = torch.ones((n,), device=device) * pos_mask
    neg_mask_batch = torch.ones((n,), device=device) * neg_mask
    zero_batch = torch.zeros((n,), device=device)

    loss_batch = true_batch * torch.square(torch.maximum(pred_batch - pos_mask_batch, zero_batch)) + \
                 (1 - true_batch) * torch.maximum(torch.square(neg_mask_batch - pred_batch), zero_batch)

    return loss_batch


def siam_loss_fn(a_batch, b_batch, p_batch):
    """
    计算该批数据中每个数据的损失值，并用其平均值作为该批数据的损失值
    :param a_batch: tensor([<a data>, ..., <a data>])
    :param b_batch: tensor([<b data>, ..., <b data>])
    :param p_batch: 标记数据对是正对还是负对，结构为：tensor([<p>, ..., <p>])
    """
    pred_batch = euclidean_distance(a_batch, b_batch)
    loss_batch = contrastive_loss(true_batch=p_batch, pred_batch=pred_batch)
    return torch.mean(loss_batch)
