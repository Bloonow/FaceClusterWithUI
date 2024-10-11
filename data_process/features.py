"""
This module supports some methods to extract feature form img_npy and generate feat_npy to return.
Note that, the feat_npy has two dimensions, image number and its feature.
"""
import numpy as np
from skimage.feature import hog


def feat_HOG(img_npy):
    feat_list = []
    for i in range(img_npy.shape[0]):
        feat = hog(img_npy[i])
        feat_list.append(feat)

    feat_npy = np.array(feat_list)
    return feat_npy


def feat_gray_delta(img_npy, delta):
    """
    将原始图片的灰度值增加或减少，来作为图片的特征
    若某点的灰度值变化超过了上下界，则取上下界作为某点处的灰度值
    """
    if len(img_npy.shape) == 3 and img_npy.shape[-1] == 3:
        factor = np.array([0.299, 0.587, 0.114])
        img_npy = np.dot(img_npy, factor)

    feat_npy = np.maximum(np.minimum(img_npy + delta, 255), 0)
    return feat_npy.reshape(feat_npy.shape[0], np.prod(feat_npy.shape[1:]))
