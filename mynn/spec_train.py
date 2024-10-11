from time import time

from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR

from mynn.generate import generate_affinity_matrix
from mynn.logger import NNLogger
from mynn.spec_ds import get_mv_dataloader
from mynn.spec_net import get_new_spec_net, spec_loss_fn
from utils.SL import load_objects, save_objects


def train_spectral_epoch_update_by_batch(spec_net, siam_net_list, mv_dloader, optimizer, view_size, lamb,
                                         n_nbrs, scale_nbrs, check_stop=lambda: False, print_scale=0.1):
    """
    每批batch进行一次反向传播和参数更新的过程
    """
    total_batch = len(mv_dloader)
    mod = max(1, int(total_batch * print_scale))

    loss = 0
    for b_idx, (data_batch_mv_list, label_batch) in enumerate(mv_dloader):
        out_batch_mv_list = spec_net(data_batch_mv_list)
        W_mv_list = []

        for vi in range(view_size):
            W = generate_affinity_matrix(siam_net_list[vi], data_batch_mv_list[vi], n_nbrs, scale_nbrs)
            W_mv_list.append(W)

        loss_in_batch = spec_loss_fn(out_batch_mv_list, W_mv_list, lamb)

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


def train_spectral(cfg, ds_from_dir, save_dir, check_stop=lambda: False):
    # 加载配置
    ds_name = cfg['dataset_name']
    view_size = cfg['view_size']
    arch = cfg['arch']
    epochs = cfg['spec_epoch_num']
    batch_size = cfg['spec_batch_size']
    lr = cfg['spec_lr']
    lr_dropout = cfg['spec_lr_dropout']
    n_nbrs = cfg['spec_n_nbrs']
    scale_nbrs = cfg['spec_scale_nbrs']
    in_dim = cfg['spec_in_dim']
    lamb = cfg['spec_lambda']

    # 生成数据集Dataloader、模型SpectralNet、优化器optimizer
    mv_dloader = get_mv_dataloader(ds_from_dir + 'train_mv.b', batch_size=batch_size)
    spec_net = get_new_spec_net(in_dim, arch, view_size)
    optimizer = RMSprop(spec_net.parameters(), lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=lr_dropout)

    # 训练SpectralNet所用的SiameseNet列表
    siam_path_list = [save_dir + 'siamese_net_view_{}.model'.format(str(vi)) for vi in range(view_size)]
    siam_net_list = load_objects(siam_path_list)

    # 所有视图一起训练，共训练epochs轮
    loss_history = []
    NNLogger(f'Spectral training, total view: {view_size}, and {epochs} epochs per view.')
    for epoch in range(1, epochs + 1):
        NNLogger(f'Epoch: {epoch} / {epochs} starting, lr = {lr_scheduler.get_last_lr()}')

        start_time = time()
        loss = train_spectral_epoch_update_by_batch(spec_net=spec_net,
                                                    siam_net_list=siam_net_list,
                                                    mv_dloader=mv_dloader,
                                                    optimizer=optimizer,
                                                    view_size=view_size,
                                                    lamb=lamb,
                                                    n_nbrs=n_nbrs,
                                                    scale_nbrs=scale_nbrs,
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
            save_objects([spec_net, loss_history],
                         [save_dir + 'spectral_net_mv.model', save_dir + 'spectral_net_mv.loss'])

        if check_stop():
            return

            # 保存模型和损失函数的变化记录
    NNLogger('Save model and loss.')
    save_objects([spec_net, loss_history],
                 [save_dir + 'spectral_net_mv.model', save_dir + 'spectral_net_mv.loss'])
