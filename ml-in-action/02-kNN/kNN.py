"""
k-近邻算法应用
"""

from numpy import *
import operator
from os import listdir
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


def create_data_set():
    """
    创建数据集
    :return group: 数据集
    :return labels: 分类标签
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 1.0]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    """
    kNN算法，分类器
    :param in_x: 测试集
    :param data_set: 训练集
    :param labels: 分类标签
    :param k: kNN算法参数，选择距离最小的k个点
    :return:
    """
    data_set_size = data_set.shape[0]  # numpy函数shape返回array每一维的长度,shape[0]即第一维的长度
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set  # numpy函数tile将数组in_x在行方向上重复data_set_size次，列方向上一次
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)  # 在第二维上进行求和（即每一列相加）
    distance = sq_distance ** 0.5  # 算出欧氏距离
    sorted_dist_indices = distance.argsort()  # 返回排序后的索引

    class_count = {}  # 记录标签出现的次数
    for i in range(k):
        vote_i_label = labels[sorted_dist_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1),
                                reverse=True)  # 字典排序itemgetter(0)根据key排序，itemgetter(1)根据value排序
    return sorted_class_count[0][0]


def file2matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split("\t")
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(list_from_line[-1])
        index += 1
    return return_mat, class_label_vector


if __name__ == '__main__':
    # 创建数据集
    g, ls = create_data_set()
    # 测试集
    test = [0, 0]
    # kNN分类
    test_class = classify0(test, g, ls, 3)
    # 打印分类结果
    print(test_class)

    file = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(file)
    print(datingDataMat)
    print(datingLabels)
