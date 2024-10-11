"""
This module supports some methods to reduce the dimension of an image feature,
so that we could improve the speed of training net.
Note that, image feature named feat_npy and the return reduction feat named rfeat_npy,
both them are two dimensions, image number and its feature.
"""

import numpy as np
from sklearn.manifold import SpectralEmbedding


def reduction_spectral_embedding(feat_npy, n_components=10, n_neighbors=10, times=5):
    n, d = feat_npy.shape
    n_components = min(n_components, d)
    n_neighbors = min(max(int(n * 0.01), n_neighbors), n)
    se = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    for _ in range(times):
        se.fit(feat_npy)
    rfeat_npy = se.fit_transform(feat_npy)

    # 规范化
    max_val = np.max(abs(rfeat_npy))
    tar_max_val = 5
    rfeat_npy = (rfeat_npy * tar_max_val / max_val).astype(np.float32)

    return rfeat_npy
