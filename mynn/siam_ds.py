import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from mynn.generate import create_pairs
from mynn.logger import NNLogger
from utils.SL import load_object


class PairsDataset(Dataset):
    """
    提供成对的数据，它每次getitem()提供的结构为：( <a data point>, <b data point>, <label> )
    使用Dataloader获得一批数据的结构为：
    ( tensor([<a data>, <>, ..., <>]), tensor([<b data>, <>, ..., <>]), tensor([<label>, <>, ..., <>]) )
    """

    def __init__(self, data_file_path, device_str='cpu', k=3):
        """
        在构造函数中从源数据文件读取数据，并生成数据的正对和负对，作为该数据集提供的数据
        :param data_file_path: 源数据文件，结构为一个元组：( <data points>, <labels> )
        其中，<data points>和<labels>都是np.ndarray类型的二维数据结构，表示 ( <numbers>, <value> )
        """
        super().__init__()
        dl = load_object(data_file_path)
        i_npy, j_npy, a_npy, b_npy, p_npy = create_pairs(raw_data=dl[0], k=k)

        self.total_num = i_npy.shape[0]

        self.a_all = torch.from_numpy(a_npy).to(device=device_str)
        self.b_all = torch.from_numpy(b_npy).to(device=device_str)
        self.p_all = torch.from_numpy(p_npy).to(device=device_str)

        NNLogger(f'PairsDataset has created {self.total_num} pairs of data.')

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        """
        :return: ( <a data point>, <b data point>, <label> )
        """
        return self.a_all[idx], self.b_all[idx], self.p_all[idx]


def get_pairs_dataloader(data_file_path, batch_size, k=3):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    pds = PairsDataset(data_file_path, device_str, k)
    return DataLoader(pds, batch_size=batch_size, shuffle=True, drop_last=True)
