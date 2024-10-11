"""
Binary file endswith .npy or .b, and its structure: ([<view_0_data_npy>, <view_1_data_npy>, ..., <>], <label_npy>)
This module supports a function to convert .npy or .b files into train_view_0.b, train_view_1.b, ..., and train_mv.b,
and into test_view_0.b, test_view_1.b, and test_mv.b, in which the test size is one third of train size randomly.
"""
import numpy as np

from utils.SL import save_object, load_object


def npy_convert_and_save(from_path, save_dir):
    # 使用.npy或.b二进制文件作为原始数据
    data_npy_list, label_npy = load_object(from_path)

    view_size = len(data_npy_list)
    total_num = label_npy.shape[0]
    t_indices = np.random.randint(0, total_num, int(total_num / 3))

    train_mv_list = []
    test_mv_list = []
    for vi in range(view_size):
        save_object((data_npy_list[vi], label_npy), path=save_dir + 'train_view_{}.b'.format(str(vi)))
        save_object((data_npy_list[vi][t_indices], label_npy[t_indices]), path=save_dir + 'test_view_{}.b'.format(str(vi)))

        train_mv_list.append(data_npy_list[vi])
        test_mv_list.append(data_npy_list[vi][t_indices])

    save_object((train_mv_list, label_npy), path=save_dir + 'train_mv.b')
    save_object((test_mv_list, label_npy[t_indices]), path=save_dir + 'test_mv.b')
