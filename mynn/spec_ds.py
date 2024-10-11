import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from mynn.logger import NNLogger
from utils.SL import load_object


class MvDataset(Dataset):
    """
    提供多个视图数据，它每次getitem()提供的结构为：( (<view_0 data>, <view_1 data>, ..., <view_m data>), <label> )
    使用Dataloader获得一批数据的结构为：
    ( [ tensor([<view_0 data>, <>, ..., <>]), tensor([<view_1 data>, <>, ..., <>]), ... ], tensor([<label>, <>, ..., <>] )
    """

    def __init__(self, data_file_path, device_str='cpu'):
        """
        :param data_file_path: 源数据文件，结构为一个元组：( [<view_0 data points>, <view_1 data points>, ..., <>], <label> )
        其中，<data points>和<labels>都是np.ndarray类型的二维数据结构，表示 (<numbers>, <value>)
        """
        super().__init__()
        dl = load_object(data_file_path)
        data_npy_mv_list, label_npy = dl

        self.view_size = len(data_npy_mv_list)
        self.total_num = label_npy.shape[0]

        self.data_all_mv_list = []
        for v_idx in range(self.view_size):
            data_npy = data_npy_mv_list[v_idx]
            self.data_all_mv_list.append(torch.from_numpy(data_npy).to(device=device_str))
        self.label_all = torch.from_numpy(label_npy).to(device=device_str)

        NNLogger(f'MvDataset has loaded {self.view_size} views, and each view has {self.total_num} data.')

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        """
        :return: ( [<view_0 data points>, <view_1 data points>, ..., <>], <label> )
        """
        data_mv_list = []
        for vi in range(self.view_size):
            data_mv_list.append(self.data_all_mv_list[vi][idx])
        return data_mv_list, self.label_all[idx]


def get_mv_dataloader(data_file_path, batch_size):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mds = MvDataset(data_file_path, device_str=device_str)
    return DataLoader(mds, batch_size=batch_size, shuffle=True, drop_last=True)
