import os
from pathlib import Path

import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread

from data_process.features import feat_gray_delta, feat_HOG
from data_process.npy import npy_convert_and_save
from data_process.original_images import images_convert
from data_process.reduction import reduction_spectral_embedding
from mynn.logger import NNLogger
from mynn.siam_train import train_siamese
from mynn.spec_train import train_spectral
from utils.SL import save_object


class TrainerThread(QThread):
    Log_Signal = pyqtSignal(str)

    def __init__(self, control_config, hyper_config):
        super(TrainerThread, self).__init__()
        self.stopping = False
        self.control_config = control_config
        self.hyper_config = hyper_config

    def run(self):
        NNLogger.register_log_fn(self.ui_log)

        self.check_dir_exists()
        if self.stopping:
            self.ui_log('Training over.')
            return

        self.data_process_and_save()
        if self.stopping:
            self.ui_log('Training over.')
            return

        self.start_training()

        NNLogger.default_log_fn()
        self.ui_log('Training over.')

    def check_dir_exists(self):
        ds_save_dir = self.control_config['ds_save_dir']
        if not Path(ds_save_dir).exists():
            os.mkdir(ds_save_dir)
        model_save_dir = self.control_config['model_save_dir']
        if not Path(model_save_dir).exists():
            os.mkdir(model_save_dir)

    def data_process_and_save(self):
        ds_from_path = self.control_config['ds_from_path']
        ds_save_dir = self.control_config['ds_save_dir']
        view_size = self.hyper_config['view_size']
        mv_ways = self.control_config['mv_ways']
        reduce_way = self.control_config['reduce_way']

        if self.stopping:
            return

        if os.path.isfile(ds_from_path) and mv_ways[0] == '.npy':
            self.ui_log('使用.npy数据集，生成数据中...')
            npy_convert_and_save(ds_from_path, ds_save_dir)
            self.ui_log('使用.npy生成数据完毕，存储完毕.')
        elif os.path.isdir(ds_from_path):
            self.ui_log('使用原始图片，读取中...')
            img_npy, label_npy = images_convert(ds_from_path)

            total_num = label_npy.shape[0]
            t_indices = np.random.randint(0, total_num, int(total_num / 3))
            train_mv_list = []
            test_mv_list = []

            self.ui_log('原始图片读取完毕，各视图数据处理中...')
            for vi in range(view_size):
                if self.stopping:
                    return
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
                if self.stopping:
                    return

                if reduce_way == 'SE: Spectral Embedding':
                    # 使用Spectral Embedding降维
                    rfeat_npy = reduction_spectral_embedding(feat_npy)
                else:
                    rfeat_npy = feat_npy

                self.ui_log(f'视图{vi}数据降维完毕，存储数据中...')

                save_object((rfeat_npy, label_npy), path=ds_save_dir + 'train_view_{}.b'.format(str(vi)))
                save_object((rfeat_npy[t_indices], label_npy[t_indices]), path=ds_save_dir + 'test_view_{}.b'.format(str(vi)))

                self.ui_log(f'视图{vi}数据存储完毕.')

                train_mv_list.append(rfeat_npy)
                test_mv_list.append(rfeat_npy[t_indices])

            save_object((train_mv_list, label_npy), path=ds_save_dir + 'train_mv.b')
            save_object((test_mv_list, label_npy[t_indices]), path=ds_save_dir + 'test_mv.b')

            self.ui_log('所有视图数据读取、提取特征、降维、存储完毕.')

    def start_training(self):
        cfg = self.hyper_config
        model_name = self.control_config['model_name']
        ds_from_dir = self.control_config['ds_save_dir']
        save_dir = self.control_config['model_save_dir']

        self.ui_log(f'模型{model_name}，开始训练...')
        if model_name == 'MvLNet: Multi-view Laplacian Network' or \
                model_name == 'Deep Spectral Representation Learning From Multi-View Data':
            train_siamese(cfg, ds_from_dir, save_dir, check_stop=lambda: self.stopping)
            if self.stopping:
                return
            train_spectral(cfg, ds_from_dir, save_dir, check_stop=lambda: self.stopping)
        elif model_name == 'MvLNet - Siamese Net':
            train_siamese(cfg, ds_from_dir, save_dir, check_stop=lambda: self.stopping)
        elif model_name == 'MvLNet - Spectral Net':
            train_spectral(cfg, ds_from_dir, save_dir, check_stop=lambda: self.stopping)
        self.ui_log(f'模型{model_name}，训练完毕，存储完毕.')

    def ui_log(self, content_str):
        self.Log_Signal.emit(content_str)

    def register_slot_fn(self, slot_fn):
        self.Log_Signal.connect(slot_fn)

    def set_stop(self, val):
        if isinstance(val, bool):
            self.stopping = val
        else:
            self.stopping = False
