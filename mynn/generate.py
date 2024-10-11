import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def create_pairs(raw_data, k):
    """
    :param k: 标识一个数据点有多少个K最近邻居
    :param raw_data: 一批数据点，结构为：np.ndarray([<data point>, <data point>, ..., <data point>])
    :return: 一个含5个元素的元组，结构为：(<i_npy>, <j_npy>, <a_npy>, <b_npy>, <p_npy>)，每个元素为一个np.ndarray
    """
    n, d = raw_data.shape
    k = min(k, int(n / 2 - 1))

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(raw_data)
    dist, indices = nbrs.kneighbors(raw_data)

    # 生成正对和负对
    i_list = []
    j_list = []
    a_list = []
    b_list = []
    p_list = []
    for raw_idx in range(n):
        # 正对，使用K最近邻居
        pos_indices = indices[raw_idx, 1:]
        # 负对，使用除了K最近邻居之外的其他数据
        neg_indices = []
        for i in range(k):
            idx = random.randint(0, n - 1)
            while (idx in pos_indices) or (idx in neg_indices):
                idx = random.randint(0, n - 1)
            neg_indices.append(idx)

        # 将正对添加到pairs中
        for p_idx in pos_indices:
            i_list.append(raw_idx)
            j_list.append(p_idx)
            a_list.append(raw_data[raw_idx])
            b_list.append(raw_data[p_idx])
            p_list.append(1)
        # 将负对添加到pairs中
        for n_idx in neg_indices:
            i_list.append(raw_idx)
            j_list.append(n_idx)
            a_list.append(raw_data[raw_idx])
            b_list.append(raw_data[n_idx])
            p_list.append(0)

    i_npy = np.array(i_list, dtype=np.int32)
    j_npy = np.array(j_list, dtype=np.int32)
    a_npy = np.array(a_list, dtype=np.float32)
    b_npy = np.array(b_list, dtype=np.float32)
    p_npy = np.array(p_list, dtype=np.int32)
    return i_npy, j_npy, a_npy, b_npy, p_npy


def generate_affinity_matrix(siam_net, data_batch, n_nbrs=5, scale_nbrs=2):
    """
    :param siam_net: SiameseNet对象
    :param data_batch: 一批数据点，结构为：tensor([<data point>, <data point>, ..., <data point>])
    :param n_nbrs: 生成近似矩阵时，指定每个数据点有几个近似的点，除此之外的近似度都为0
    :param scale_nbrs: 用来求scale用的n_nbrs
    """
    n, d = data_batch.size()
    device = data_batch.device
    n_nbrs = min(n_nbrs, n)

    out_batch = siam_net(data_batch)

    # 计算用作 sigma = scale 的值
    data_batch_on_cpu = out_batch.detach().cpu()
    nbrs = NearestNeighbors(n_neighbors=scale_nbrs).fit(data_batch_on_cpu)
    dist, _ = nbrs.kneighbors(data_batch_on_cpu)
    # scale的值受数据数目影响较大，在不同数目的数据批上求K最近距离的均值，得到的值差距较大；
    # 数据越少，则它们的K最近邻居越不正确，距离dist就相对越大，得到的scale就越大，从而近似矩阵W值就越大
    # 为防止这种情况发生，可以让DataLoader将最后加载的一批数目小于batch_size的数据丢弃掉
    scale = np.median(dist[:, scale_nbrs - 1])

    # 空间换时间
    A = torch.empty((n, n, d)).cuda()
    B = torch.empty((n, n, d)).cuda()
    for i in range(n):
        A[i, :] = out_batch[i]
        B[:, i] = out_batch[i]
    Dy = torch.sum((A - B) ** 2, dim=2)

    # 在距离矩阵D中，与自身的距离是0，故需要n_nbrs + 1
    # top_D, top_indices = torch.topk(Dy, k=n_nbrs + 1, largest=False)
    # W = torch.zeros((n, n), device=device)
    # for idx in range(n):
    #     W[idx, top_indices[idx]] = torch.exp(-top_D[idx] / (2 * scale ** 2))

    W = torch.exp(-Dy / (2 * scale ** 2))
    return W
