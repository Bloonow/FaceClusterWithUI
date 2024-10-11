"""
This module supports a function to read images and convert them into img_npy and label_npy,
which the image filename should startswith 00_, 01_, ... so that we could identify the label of them.
"""
import os

import numpy as np
from skimage.io import imread


def images_convert(from_dir):
    """
    加载原始图片和它的标识
    图片的R、G、B以及灰度Gray值的范围是：0 - 255
    """
    items = os.listdir(from_dir)
    img_list = []
    label_list = []
    for item in items:
        path = from_dir + item
        img = imread(path)
        img.astype(np.int16)
        img_list.append(img)
        label = int(item[:item.find('_')])
        label_list.append(label)

    img_npy = np.array(img_list)
    label_npy = np.array(label_list)
    return img_npy, label_npy
