"""
Multi-view clustering and evaluation
"""

import numpy as np
import sklearn.metrics as metrics
from munkres import Munkres
from sklearn.cluster import KMeans

from mynn.logger import NNLogger


def clustering(mv_data_batch, label_batch):
    """
    :param mv_data_batch: 模型输出的多视图列表，元素为numpy.ndarray类型，结构为：
                [array([<data>, <data>, ..., <data>]), ..., ,array([<another view data>, <>, ..., <>])]
    :param label_batch: 多视图数据的真实标记，一个numpy.ndarray数值，结构为：array([<label>, <label>, ..., <label>])
    """
    NNLogger('Clustering starting...')
    # 获得聚类的簇数
    n_cluster = np.size(np.unique(label_batch))
    # 将每个数据点在多个视图的数据连接在一起：array([<view1, view2, ...>, <view1, view2, ...>, ..., <view1, view2, ...>])
    data_batch = np.concatenate(mv_data_batch[:], axis=1)
    # 如果簇标签是从1开始的，则将其改为从0开始
    if np.min(label_batch) == 1:
        label_batch = label_batch - 1
    if label_batch.dtype != np.int32:
        label_batch = label_batch.astype(np.int32)

    # 获得聚类分配，因为聚类的标签可能与真实标签不是采用相同的值表示，如聚类分配使用0表示cat，而真实值使用1表示cat
    # 故需要对其进行处理，找到尽可能正确的对应关系
    kmeans_assignment = get_cluster_assignment(data_batch, n_cluster, n_init=10)
    pred_batch = get_prediction(cluster_assignment=kmeans_assignment, true_batch=label_batch, n_cluster=n_cluster)

    # 计算聚类结果的各项指标
    cluster_metric(pred_batch=pred_batch, true_batch=label_batch)
    classification_metric(pred_batch=pred_batch, true_batch=label_batch)

    return pred_batch


def get_cluster_assignment(data_batch, n_clusters, **kwargs):
    """
    获得聚类分配
    :return: 该批数据data_batch聚类后的簇标签，结构为：array([<label>, <label>, ..., <label>])
    """
    kmeans_obj = KMeans(n_clusters, **kwargs)
    for i in range(10):
        kmeans_obj.fit(data_batch)
    cluster_assignment = kmeans_obj.predict(data_batch)
    return cluster_assignment


def get_prediction(cluster_assignment, true_batch, n_cluster):
    """
    根据聚类分配和ground-truth标签获得预测结果
    :param cluster_assignment: 一批数据聚类后的簇标签，结构为：array([<label>, <label>, ..., <label>])
    :param true_batch: 同一批数据真实的ground-truth标签，结构为：array([<truth>, <truth>, ..., <truth>])
    """
    # 根据y_true和p_pred计算混淆矩阵
    # confusion_matrix[i, j] 表示真实属于i类，却被分为j类的个数
    confusion_matrix = metrics.confusion_matrix(y_true=true_batch, y_pred=cluster_assignment, labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_cluster)

    # 获得原始聚类分类的索引到正确索引之间的映射，如：[(0, 3), (1, 2), (2, 0), (3, 1)]
    map_indices_tuple = Munkres().compute(cost_matrix)
    map_indices = np.zeros((n_cluster,), dtype=np.int32)
    for i in range(n_cluster):
        map_indices[i] = map_indices_tuple[i][1]

    if np.min(cluster_assignment) != 0:
        cluster_assignment = cluster_assignment - np.min(cluster_assignment)

    pred_batch = map_indices[cluster_assignment]
    return pred_batch


def calculate_cost_matrix(confusion_matrix, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i, j] 表示把第i类数据，聚类为第j类所需要的代价
    # 代价越小，相应的聚类就应该越正确，即第i类实际上对应的是第j类
    for j in range(n_clusters):
        # 计算被聚类为第j簇数据的数目，即有多少数据被聚类为第j簇，即在j簇中样例的数目
        cluster_j_num = np.sum(confusion_matrix[:, j])
        for i in range(n_clusters):
            # confusion_matrix[i, j] 表示真实属于i类，却被分为j类的个；在被分为第j类中，样本来自类i的数目
            # cost_matrix[j, i] 表示在被分为类j的样本中，不是来自第i类的样本数目，
            # 若 j != i，则值越大表示分类越正确；若 j == i，则值越小表示分类越正确
            cost_matrix[j, i] = cluster_j_num - confusion_matrix[i, j]
    return cost_matrix


def cluster_metric(pred_batch, true_batch, decimals=4):
    # AMI
    ami = metrics.adjusted_mutual_info_score(true_batch, pred_batch)
    ami = np.round(ami, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(true_batch, pred_batch)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(true_batch, pred_batch)
    ari = np.round(ari, decimals)

    NNLogger(f'AMI: {ami}, NMI: {nmi}, ARI: {ari}')


def classification_metric(pred_batch, true_batch, decimals=4, average='macro'):
    # ACC
    accuracy = metrics.accuracy_score(true_batch, pred_batch)
    accuracy = np.round(accuracy, decimals)
    # precision
    precision = metrics.precision_score(true_batch, pred_batch, average=average)
    precision = np.round(precision, decimals)
    # recall
    recall = metrics.recall_score(true_batch, pred_batch, average=average)
    recall = np.round(recall, decimals)
    # F-score
    f_score = metrics.f1_score(true_batch, pred_batch, average=average)
    f_score = np.round(f_score, decimals)

    NNLogger(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, f_measure: {f_score}')
