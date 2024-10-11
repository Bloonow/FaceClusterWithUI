from time import time

import torch

from mynn.logger import NNLogger
from mynn.siam_ds import get_pairs_dataloader
from mynn.siam_net import get_new_siam_net, siam_loss_fn
from utils.SL import save_objects


def train_siamese_epoch_update_by_batch(siam_net, dloader, optimizer, check_stop=lambda: False, print_scale=0.1):
    """
    每批batch进行一次反向传播和参数更新的过程
    """
    total_batch = len(dloader)
    mod = max(1, int(total_batch * print_scale))

    loss = 0
    for b_idx, (a_batch, b_batch, p_batch) in enumerate(dloader):
        # 数据的正对和负对应该是全局的，而不是在每批数据中创建
        a_out_batch, b_out_batch = siam_net(a_batch), siam_net(b_batch)

        loss_in_batch = siam_loss_fn(a_out_batch, b_out_batch, p_batch=p_batch)

        # 打印信息
        b_no = b_idx + 1
        if b_no % mod == 0 or b_no == total_batch:
            NNLogger(f'Batch: {b_no} / {total_batch}, loss = {loss_in_batch.item()}')

        # 反向传播
        optimizer.zero_grad()
        loss_in_batch.backward()
        optimizer.step()

        loss += loss_in_batch.item()

        if check_stop():
            return

    loss /= total_batch
    return loss


def train_siamese(cfg, ds_from_dir, save_dir, check_stop=lambda: False):
    # 加载配置
    ds_name = cfg['dataset_name']
    view_size = cfg['view_size']
    arch = cfg['arch']
    epochs = cfg['siam_epoch_num']
    batch_size = cfg['siam_batch_size']
    lr_start = cfg['siam_lr']
    lr_dropout = cfg['siam_lr_dropout']
    k_nbrs = cfg['siam_k_nbrs']
    in_dim = cfg['siam_in_dim']

    train_dataset_path_pat = ds_from_dir + 'train_view_{}.b'

    # 每个视图单独训练，每个视图训练epochs轮
    NNLogger(f'SiameseNet training, total view: {view_size}, and {epochs} epochs per view.')

    for vi in range(view_size):
        NNLogger(f'View no: {vi + 1}')

        # 加载这个视图的数据集
        train_dataset_path = train_dataset_path_pat.format(vi)
        dloader = get_pairs_dataloader(train_dataset_path, batch_size, k_nbrs)

        # 构造这个视图的SiameseNet及其优化器
        siam_net = get_new_siam_net(in_dim, arch)
        optimizer = torch.optim.RMSprop(siam_net.parameters(), lr=lr_start)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_dropout)

        loss_history = []
        for epoch in range(1, epochs + 1):
            NNLogger(f'Epoch: {epoch} / {epochs} starting, lr = {lr_scheduler.get_last_lr()}')

            start_time = time()
            loss = train_siamese_epoch_update_by_batch(siam_net=siam_net,
                                                       dloader=dloader,
                                                       optimizer=optimizer,
                                                       check_stop=check_stop)
            end_time = time()
            used_time = '{:.2f}'.format(end_time - start_time)
            NNLogger(f'Epoch: {epoch} / {epochs}, used time: {used_time} s, and its loss = {loss}')

            # 调整lr，指数衰减
            lr_scheduler.step()

            loss_history.append(loss)

            if epoch % 10 == 0:
                # 每10轮保存一次模型
                # 保存模型和损失函数的变化记录
                NNLogger('Save model and loss.')
                save_objects([siam_net, loss_history],
                             [save_dir + 'siamese_net_view_{}.model'.format(str(vi)),
                              save_dir + 'siamese_net_view_{}.loss'.format(str(vi))])

            if check_stop():
                return

        # 保存模型和损失函数的变化记录
        NNLogger('Save model and loss.')
        save_objects([siam_net, loss_history],
                     [save_dir + 'siamese_net_view_{}.model'.format(str(vi)),
                      save_dir + 'siamese_net_view_{}.loss'.format(str(vi))])
