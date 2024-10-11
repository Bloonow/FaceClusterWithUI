import os

import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal

from data_process.features import feat_gray_delta, feat_HOG
from data_process.original_images import images_convert
from data_process.reduction import reduction_spectral_embedding
from mynn.cluster import clustering
from mynn.logger import NNLogger
from utils.SL import load_object


class ClusterThread(QThread):
    Log_Signal = pyqtSignal(str)
    Show_Signal = pyqtSignal(np.ndarray)

    def __init__(self, control_config):
        super(ClusterThread, self).__init__()
        self.stopping = False
        self.control_config = control_config

        self.model_net = load_object(control_config['use_model_path'])

    def run(self):
        NNLogger.register_log_fn(self.ui_log)

        test_mv_list, label_npy = self.data_process()
        out_mv_list = self.model_net(test_mv_list)
        out_list = [data.cpu().detach().numpy() for data in out_mv_list]
        pred_npy = clustering(out_list, label_npy)

        self.ui_log(str(pred_npy))

        ds_from_path = self.control_config['ds_from_path']
        if ds_from_path != '' and os.path.isdir(ds_from_path):
            self.show_prediction(pred_npy)

        NNLogger.default_log_fn()
        self.ui_log('Clustering over.')

    def data_process(self):
        ds_from_path = self.control_config['ds_from_path']
        reduce_way = self.control_config['reduce_way']
        mv_ways = self.control_config['mv_ways']
        view_size = len(mv_ways)

        if os.path.isfile(ds_from_path) and mv_ways[0] == '.npy':
            self.ui_log('使用test_mv.b测试集，加载中...')
            test_mv_list, label_npy = load_object(ds_from_path)
            self.ui_log('测试集test_mv.b加载完毕.')
        else:
            #  os.path.isdir(ds_from_path) == True
            self.ui_log('使用原始图片，读取中...')
            img_npy, label_npy = images_convert(ds_from_path)

            test_mv_list = []
            self.ui_log('原始图片读取完毕，各视图数据处理中...')
            for vi in range(view_size):
                self.ui_log(f'视图{vi}提取图片特征中...')
                way = mv_ways[vi]
                if way == 'brightness_gray':
                    feat_npy = img_npy.reshape(img_npy.shape[0], np.prod(img_npy.shape[1:]))
                elif 'brightness_plus_' in way or 'brightness_minus_' in way:
                    delta = int(way[way.rfind('_') + 1:])
                    feat_npy = feat_gray_delta(img_npy, delta)
                elif way == 'feat_HOG':
                    feat_npy = feat_HOG(img_npy)
                else:
                    feat_npy = img_npy.reshape(img_npy.shape[0], np.prod(img_npy.shape[1:]))

                self.ui_log(f'视图{vi}提取图片特征完毕，数据降维中...')

                if reduce_way == 'SE: Spectral Embedding':
                    # 使用Spectral Embedding降维
                    rfeat_npy = reduction_spectral_embedding(feat_npy)
                else:
                    rfeat_npy = feat_npy

                self.ui_log(f'视图{vi}数据降维完毕.')

                test_mv_list.append(rfeat_npy)

            self.ui_log('所有视图数据读取、提取特征、降维完毕.')

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        test_mv_list = [torch.from_numpy(data).to(device_str) for data in test_mv_list]

        return test_mv_list, label_npy

    def ui_log(self, content_str):
        self.Log_Signal.emit(content_str)

    def register_log_slot_fn(self, slot_fn):
        self.Log_Signal.connect(slot_fn)

    def show_prediction(self, pred_npy):
        self.Show_Signal.emit(pred_npy)

    def register_show_slot_fn(self, slot_fn):
        self.Show_Signal.connect(slot_fn)
