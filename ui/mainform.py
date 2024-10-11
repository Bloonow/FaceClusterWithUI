import _pickle as pickle
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog, QHBoxLayout, QLabel, QVBoxLayout

from ui.cluster_thread import ClusterThread
from ui.logform import LogForm
from ui.train_thread import TrainerThread
from ui_autogen.mainwindow import Ui_MainWindow
from utils.SL import load_object


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainForm, self).__init__(parent)
        self.setupUi(self)

        self.setWindowTitle('朱泽央-毕业设计')

        # 设置显示图片聚类结果的控件
        self.gallery_layout = QVBoxLayout(self.scrollAreaWidget_show_images)
        self.gallery_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.pred_npy = None

        # 为各个控件设置相应的槽函数
        self.set_slots_in_train_tab()
        self.set_slots_in_cluster_tab()

    def set_slots_in_train_tab(self):
        self.checkBox_use_npy.stateChanged.connect(self.checkBox_use_npy_slot_fn)
        self.checkBox_use_images.stateChanged.connect(self.checkBox_use_images_slot_fn)
        self.pushButton_select_npy_or_images.clicked.connect(self.pushButton_select_npy_or_images_slot_fn)

        self.lineEdit_ds_name.textChanged.connect(self.lineEdit_ds_name_slot_fn)
        self.pushButton_select_ds_save_path.clicked.connect(self.pushButton_select_ds_save_path_slot_fn)
        self.pushButton_select_model_save_path.clicked.connect(self.pushButton_select_model_save_path_slot_fn)
        self.pushButton_select_config_load_path.clicked.connect(self.pushButton_select_config_load_path_slot_fn)

        self.pushButton_start_training.clicked.connect(self.pushButton_start_training_slot_fn)

    def checkBox_use_npy_slot_fn(self):
        checked = self.checkBox_use_npy.isChecked()
        self.checkBox_use_images.setChecked(not checked)

        self.tabWidget_mv.setEnabled(not checked)
        self.comboBox_select_reduction_ways.setEnabled(not checked)

        self.clear_dataset_preview()

    def checkBox_use_images_slot_fn(self):
        checked = self.checkBox_use_images.isChecked()
        self.checkBox_use_npy.setChecked(not checked)

        self.tabWidget_mv.setEnabled(checked)
        self.comboBox_select_reduction_ways.setEnabled(checked)

        self.clear_dataset_preview()

    def clear_dataset_preview(self):
        self.label_show_ds_from_path.setText('')
        self.update_dataset_preview()

    def update_dataset_preview(self):
        path = self.label_show_ds_from_path.text()
        if os.path.isfile(path):
            # 预加载.npy类型的二进制
            show_info = load_object(path)
        elif os.path.isdir(path):
            show_info = os.listdir(path)
        else:
            show_info = ''
        self.textBrowser_show_ds_preview.setText(str(show_info))

    def pushButton_select_npy_or_images_slot_fn(self):
        path = None
        if self.checkBox_use_npy.isChecked():
            path, file_format = QFileDialog.getOpenFileName(self, '选择数据集', './', 'Binary files (*.npy *.b);;All files (*.*)')
        elif self.checkBox_use_images.isChecked():
            path = QFileDialog.getExistingDirectory(self, '选择文件夹', './')
            if path is not None and path != '':
                path += '/'
        else:
            # 使用.npy和使用原始图片都没有选择，提示错误
            QMessageBox.warning(self, '选择错误', '请选择使用.npy数据集或使用原始图片数据集')
        if path is not None:
            self.label_show_ds_from_path.setText(path)
            self.update_dataset_preview()

    def lineEdit_ds_name_slot_fn(self):
        ds_name = self.lineEdit_ds_name.text()
        if ds_name is None or ds_name == '':
            ds_name = 'Dataset_Name'
        self.label_show_ds_save_path.setText('./' + ds_name + '/')
        self.label_show_model_save_path.setText('./' + ds_name + '/')

    def get_select_save_path_and_ds_name(self):
        path = QFileDialog.getExistingDirectory(self, '选择文件夹', './')
        if path is not None and path != '':
            path += '/'
        else:
            path = './'
        ds_name = self.lineEdit_ds_name.text()
        if ds_name is None or ds_name == '':
            ds_name = 'Dataset_Name'
        return path, ds_name

    def pushButton_select_ds_save_path_slot_fn(self):
        path, ds_name = self.get_select_save_path_and_ds_name()
        self.label_show_ds_save_path.setText(path + ds_name + '/')

    def pushButton_select_model_save_path_slot_fn(self):
        path, ds_name = self.get_select_save_path_and_ds_name()
        self.label_show_model_save_path.setText(path + ds_name + '/')

    def pushButton_select_config_load_path_slot_fn(self):
        path, file_format = QFileDialog.getOpenFileName(self, '选择配置文件', './', 'Yaml files (*.yaml)')
        self.label_show_config_load_path.setText(path)
        can_config = True if path is None or path == '' else False
        # 根据是否使用配置文件，将设置自定义参数的控件disable或enable
        self.spinBox_eopchs.setEnabled(can_config)
        self.spinBox_batch_size.setEnabled(can_config)
        self.doubleSpinBox_lr.setEnabled(can_config)
        self.doubleSpinBox_lr_dropout.setEnabled(can_config)
        self.plainTextEdit_hyper_params.setEnabled(can_config)

    def infer_mv_ways(self):
        mv_ways = []
        if self.checkBox_use_npy.isChecked():
            mv_ways.append('.npy')
            try:
                path = self.label_show_ds_from_path.text()
                with open(path, mode='rb') as f:
                    dl = pickle.load(f)
                view_size = len(dl[0])
            except IOError:
                view_size = 1
        elif self.checkBox_use_images.isChecked():
            cur_tab_idx = self.tabWidget_mv.currentIndex()
            if cur_tab_idx == 0:
                mv_ways.append('color_r') if self.checkBox_color_r.isChecked() else 0
                mv_ways.append('color_g') if self.checkBox_color_g.isChecked() else 0
                mv_ways.append('color_b') if self.checkBox_color_b.isChecked() else 0
                mv_ways.append('color_gray') if self.checkBox_color_gray.isChecked() else 0
                view_size = len(mv_ways)
            elif cur_tab_idx == 1:
                mv_ways.append('brightness_gray') if self.checkBox_original_gray.isChecked() else 0
                scatter_num = self.spinBox_gray_scatter_num.value()
                scatter_val = self.spinBox_gray_scatter_value.value()
                for i in range(1, scatter_num + 1):
                    mv_ways.append('brightness_plus_{}'.format(str(i * scatter_val)))
                    mv_ways.append('brightness_minus_{}'.format(str(i * scatter_val)))
                view_size = len(mv_ways)
            elif cur_tab_idx == 2:
                mv_ways.append('feat_HOG') if self.checkBox_feat_HOG.isChecked() else 0
                mv_ways.append('feat_GIST') if self.checkBox_feat_GIST.isChecked() else 0
                mv_ways.append('feat_LBP') if self.checkBox_feat_LBP.isChecked() else 0
                mv_ways.append('feat_Gabor') if self.checkBox_feat_Gabor.isChecked() else 0
                mv_ways.append('feat_CENTRIST') if self.checkBox_feat_CENTRIST.isChecked() else 0
                view_size = len(mv_ways)
            else:
                view_size = 1
        else:
            view_size = 1
        return view_size, mv_ways

    def collect_control_config(self):
        model_name = self.comboBox_select_model.currentText()
        if model_name == 'None':
            QMessageBox.warning(self, '警告', '请选择训练模型')
            return {}
        ds_from_path = self.label_show_ds_from_path.text()
        if ds_from_path == '':
            QMessageBox.warning(self, '警告', '请选择数据集')
            return {}
        ds_save_dir = self.label_show_ds_save_path.text()
        model_save_dir = self.label_show_model_save_path.text()
        reduce_way = self.comboBox_select_reduction_ways.currentText()

        # 本次训练所用到的各种控制参数
        control_config = {
            'model_name': model_name,
            'ds_from_path': ds_from_path,
            'ds_save_dir': ds_save_dir,
            'model_save_dir': model_save_dir,
            'reduce_way': reduce_way
        }
        return control_config

    def collect_hyper_parameters(self):
        config_load_path = self.label_show_config_load_path.text()
        if config_load_path is not None and config_load_path != '':
            with open(config_load_path, mode='r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        else:
            ds_name = self.lineEdit_ds_name.text()
            if ds_name is None or ds_name == '':
                ds_name = 'Dataset_Name'
            cfg = {
                'dataset_name': ds_name,
                'siam_epoch_num': int(self.spinBox_eopchs.value()),
                'spec_epoch_num': int(self.spinBox_eopchs.value()),
                'siam_batch_size': int(self.spinBox_batch_size.value()),
                'spec_batch_size': int(self.spinBox_batch_size.value()),
                'siam_lr': float(self.doubleSpinBox_lr.value()),
                'spec_lr': float(self.doubleSpinBox_lr.value()),
                'siam_lr_dropout': float(self.doubleSpinBox_lr_dropout.value()),
                'spec_lr_dropout': float(self.doubleSpinBox_lr_dropout.value()),
                'siam_k_nbrs': 3,
                'spec_n_nbrs': 5,
                'spec_scale_nbrs': 2,
                'siam_in_dim': 10,
                'spec_in_dim': 10,
                'spec_lambda': 0.001,
                'arch': [
                    {'layer': 'linear', 'size': 2048, 'activation': 'relu'},
                    {'layer': 'linear', 'size': 1024, 'activation': 'relu'},
                    {'layer': 'linear', 'size': 512, 'activation': 'relu'},
                    {'layer': 'linear', 'size': 10, 'activation': 'relu'}
                ]
            }
            extra_cfg_str = self.plainTextEdit_hyper_params.toPlainText()
            if extra_cfg_str is not None and extra_cfg_str != '':
                extra_cfg = {}
                extra_param_str_list = extra_cfg_str.split('\n')
                for kv in extra_param_str_list:
                    key, val_str = [item.strip() for item in kv.split(':')]
                    if val_str.isdigit():
                        val = int(val_str)
                    else:
                        try:
                            val = float(val_str)
                        except ValueError:
                            val = val_str
                    extra_cfg.update({key: val})
                cfg.update(extra_cfg)
        return cfg

    def pushButton_start_training_slot_fn(self):
        view_size, mv_ways = self.infer_mv_ways()

        # 收集本次训练所用的各种控制参数
        control_config = self.collect_control_config()
        if control_config == {}:
            return
        control_config.update({'mv_ways': mv_ways})

        # 收集本次训练所用的各种超参数
        hyper_config = self.collect_hyper_parameters()
        hyper_config.update({'view_size': view_size})

        self.lf = LogForm()
        self.trainer_thread = TrainerThread(control_config, hyper_config)
        self.lf.register_callback(self.trainer_thread.set_stop)
        self.trainer_thread.register_slot_fn(self.lf.append_log)

        self.lf.show()
        self.trainer_thread.start()

    """######################################################################"""

    def set_slots_in_cluster_tab(self):
        self.checkBox_use_npy_2.stateChanged.connect(self.checkBox_use_npy_2_slot_fn)
        self.checkBox_use_images_2.stateChanged.connect(self.checkBox_use_images_2_slot_fn)
        self.pushButton_select_npy_or_images_2.clicked.connect(self.pushButton_select_npy_or_images_2_slot_fn)

        self.pushButton_select_use_model_path.clicked.connect(self.pushButton_select_use_model_path_slot_fn)
        self.pushButton_start_clustering.clicked.connect(self.pushButton_start_clustering_slot_fn)
        self.pushButton_select_result_save_dir.clicked.connect(self.pushButton_select_result_save_dir_slot_fn)
        self.pushButton_save_result.clicked.connect(self.pushButton_save_result_slot_fn)

    def checkBox_use_npy_2_slot_fn(self):
        checked = self.checkBox_use_npy_2.isChecked()
        self.checkBox_use_images_2.setChecked(not checked)

        self.tabWidget_mv_2.setEnabled(not checked)
        self.comboBox_select_reduction_ways_2.setEnabled(not checked)

        self.scrollArea_cluster_result.setEnabled(not checked)
        self.pushButton_select_result_save_dir.setEnabled(not checked)
        self.pushButton_save_result.setEnabled(not checked)

        self.clear_dataset_preview_2()

    def checkBox_use_images_2_slot_fn(self):
        checked = self.checkBox_use_images_2.isChecked()
        self.checkBox_use_npy_2.setChecked(not checked)

        self.tabWidget_mv_2.setEnabled(checked)
        self.comboBox_select_reduction_ways_2.setEnabled(checked)
        self.scrollArea_cluster_result.setEnabled(checked)
        self.pushButton_select_result_save_dir.setEnabled(checked)
        self.pushButton_save_result.setEnabled(checked)

        self.clear_dataset_preview_2()

    def clear_dataset_preview_2(self):
        self.label_show_ds_from_path_2.setText('')
        self.update_dataset_preview_2()

    def update_dataset_preview_2(self):
        path = self.label_show_ds_from_path_2.text()
        if os.path.isfile(path):
            # 预加载.npy类型的二进制
            show_info = load_object(path)
        elif os.path.isdir(path):
            show_info = os.listdir(path)
        else:
            show_info = ''
        self.textBrowser_show_ds_preview_2.setText(str(show_info))

    def pushButton_select_npy_or_images_2_slot_fn(self):
        path = None
        if self.checkBox_use_npy_2.isChecked():
            path, file_format = QFileDialog.getOpenFileName(self, '选择测试集', './', 'Binary files (*.npy *.b);;All files (*.*)')
        elif self.checkBox_use_images_2.isChecked():
            path = QFileDialog.getExistingDirectory(self, '选择文件夹', './')
            if path is not None and path != '':
                path += '/'
        else:
            # 使用.npy和使用原始图片都没有选择，提示错误
            QMessageBox.warning(self, '选择错误', '请选择使用.npy数据集或使用原始图片数据集')
        if path is not None:
            self.label_show_ds_from_path_2.setText(path)
            self.update_dataset_preview_2()

    def pushButton_select_use_model_path_slot_fn(self):
        path, file_format = QFileDialog.getOpenFileName(self, '选择测试集', './', 'Model files (*.model)')
        if path is not None and path != '':
            self.label_show_use_model_path.setText(path)
        else:
            self.label_show_use_model_path.setText('')

    def infer_mv_ways_2(self):
        mv_ways = []
        if self.checkBox_use_npy_2.isChecked():
            mv_ways.append('.npy')
            try:
                path = self.label_show_ds_from_path_2.text()
                with open(path, mode='rb') as f:
                    dl = pickle.load(f)
                view_size = len(dl[0])
            except IOError:
                view_size = 1
        elif self.checkBox_use_images_2.isChecked():
            cur_tab_idx = self.tabWidget_mv_2.currentIndex()
            if cur_tab_idx == 0:
                mv_ways.append('color_r') if self.checkBox_color_r_2.isChecked() else 0
                mv_ways.append('color_g') if self.checkBox_color_g_2.isChecked() else 0
                mv_ways.append('color_b') if self.checkBox_color_b_2.isChecked() else 0
                mv_ways.append('color_gray') if self.checkBox_color_gray_2.isChecked() else 0
                view_size = len(mv_ways)
            elif cur_tab_idx == 1:
                mv_ways.append('brightness_gray') if self.checkBox_original_gray_2.isChecked() else 0
                scatter_num = self.spinBox_gray_scatter_num_2.value()
                scatter_val = self.spinBox_gray_scatter_value_2.value()
                for i in range(1, scatter_num + 1):
                    mv_ways.append('brightness_plus_{}'.format(str(i * scatter_val)))
                    mv_ways.append('brightness_minus_{}'.format(str(i * scatter_val)))
                view_size = len(mv_ways)
            elif cur_tab_idx == 2:
                mv_ways.append('feat_HOG') if self.checkBox_feat_HOG_2.isChecked() else 0
                mv_ways.append('feat_GIST') if self.checkBox_feat_GIST_2.isChecked() else 0
                mv_ways.append('feat_LBP') if self.checkBox_feat_LBP_2.isChecked() else 0
                mv_ways.append('feat_Gabor') if self.checkBox_feat_Gabor_2.isChecked() else 0
                mv_ways.append('feat_CENTRIST') if self.checkBox_feat_CENTRIST_2.isChecked() else 0
                view_size = len(mv_ways)
            else:
                view_size = 1
        else:
            view_size = 1
        return view_size, mv_ways

    def collect_control_config_2(self):
        use_model_path = self.label_show_use_model_path.text()
        if use_model_path == '':
            QMessageBox.warning(self, '警告', '请选择使用的模型')
            return {}

        ds_from_path = self.label_show_ds_from_path_2.text()
        if ds_from_path == '':
            QMessageBox.warning(self, '警告', '请选择测试集')
            return {}
        reduce_way = self.comboBox_select_reduction_ways_2.currentText()

        # 本次训练所用到的各种控制参数
        control_config = {
            'use_model_path': use_model_path,
            'ds_from_path': ds_from_path,
            'reduce_way': reduce_way
        }
        return control_config

    def pushButton_start_clustering_slot_fn(self):
        view_size, mv_ways = self.infer_mv_ways_2()

        # 收集本次聚类所用的各种控制参数
        control_config = self.collect_control_config_2()
        if control_config == {}:
            return
        control_config.update({'mv_ways': mv_ways})

        self.cluster_thread = ClusterThread(control_config)
        self.cluster_thread.register_log_slot_fn(self.cluster_tab_append_log_slot_fn)
        self.cluster_thread.register_show_slot_fn(self.cluster_tab_show_result_images_slot_fn)
        self.cluster_thread.start()

    def cluster_tab_append_log_slot_fn(self, content_str):
        self.textBrowser_show_cluster_result.append(content_str)

    def cluster_tab_show_result_images_slot_fn(self, pred_npy):
        # 如果数据来源是一个目录，则表示原始图片的目录
        ds_from_path = self.label_show_ds_from_path_2.text()
        if ds_from_path == '' or os.path.isfile(ds_from_path):
            QMessageBox.warning(self, '警告', '请选择正确测试集')
            return

        self.pred_npy = pred_npy

        items = os.listdir(ds_from_path)
        paths = [ds_from_path + item for item in items]

        unique_npy = np.unique(pred_npy)
        n_cluster = np.size(unique_npy)

        for row_idx in range(n_cluster):
            row_val = unique_npy[row_idx]
            indices = np.argwhere(pred_npy == row_val).squeeze()
            max_num = min(10, indices.shape[0])

            row_layout = QHBoxLayout(self.scrollAreaWidget_show_images)
            row_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
            label_show_val = QLabel(self.scrollAreaWidget_show_images)
            label_show_val.setText(f'类别{row_idx}：')
            row_layout.addWidget(label_show_val)

            for i in range(max_num):
                path = paths[indices[i]]
                label_show_image = QLabel(self.scrollAreaWidget_show_images)
                label_show_image.setMaximumSize(100, 100)
                label_show_image.setPixmap(QtGui.QPixmap(path))
                label_show_image.setScaledContents(True)
                row_layout.addWidget(label_show_image)

            self.gallery_layout.addLayout(row_layout)

    def pushButton_select_result_save_dir_slot_fn(self):
        path = QFileDialog.getExistingDirectory(self, '选择文件夹', './')
        if path is not None and path != '':
            path += '/'
        else:
            path = './'
        self.label_show_result_save_dir.setText(path)

    def pushButton_save_result_slot_fn(self):
        # 如果数据来源是一个目录，则表示原始图片的目录
        ds_from_path = self.label_show_ds_from_path_2.text()
        if ds_from_path == '' or os.path.isfile(ds_from_path):
            QMessageBox.warning(self, '警告', '请选择正确测试集')
            return

        if self.pred_npy is None:
            QMessageBox.warning(self, '警告', '请先进行聚类')
            return

        save_dir = self.label_show_result_save_dir.text()
        if not Path(save_dir).exists():
            os.mkdir(save_dir)

        items = os.listdir(ds_from_path)

        pred_npy = self.pred_npy
        unique_npy = np.unique(pred_npy)
        n_cluster = np.size(unique_npy)

        for l_idx in range(n_cluster):
            l_val = unique_npy[l_idx]
            indices = np.argwhere(pred_npy == l_val).squeeze()

            l_dir = save_dir + str(l_val) + '/'
            if not Path(l_dir).exists():
                os.mkdir(l_dir)
            for img_idx in indices:
                item = items[img_idx]
                src_path = ds_from_path + item
                tar_path = l_dir + item
                shutil.copyfile(src_path, tar_path)

            self.textBrowser_show_cluster_result.append(f'类别{l_val}已保存到{str(l_dir)}文件中，继续保存其他类别...')

        self.textBrowser_show_cluster_result.append('所有类别已保存完成.')
        QMessageBox.information(self, '信息', '所有类别已保存完成.')

