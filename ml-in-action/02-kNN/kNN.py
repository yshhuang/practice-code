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
    sq_distance = sq_diff_mat.sum(axis=1)  # 在第二维上进行求和（即每一列相加,axis只能是0或1
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
    """
    读取文件并转换为矩阵
    :param filename: 文件路径
    :return: 转换后的特征矩阵
    :return: 分类label向量
    """
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()  # str.strip([chars])用于移除字符串头尾的指定字符序列，默认为空白字符
        list_from_line = line.split("\t")
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append({
                                      "didntLike": 1,
                                      "smallDoses": 2,
                                      "largeDoses": 3
                                  }[list_from_line[-1]])
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    """
    数据归一化
    :param data_set:原特征矩阵
    :return norm_data_set: 归一化后的特征矩阵
    :return ranges: 每一列的数据最大值最小值之差
    :return min_vals: 每一列的最小值
    """
    min_vals = data_set.min(0)  # min(0)以行为维度，即计算每一列的最小值
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))  # numpy 的"/"表示矩阵的每一个元素相除，矩阵的除法用linalg.solve(matA,matB)
    return norm_data_set, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.10  # 取百分之十作为测试集
    dating_data_mat, dating_labels = file2matrix("datingTestSet.txt")
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.00
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 7)

        if classifier_result != dating_labels[i]:
            print("分类结果：%d\t真实类别：%d" % (classifier_result, dating_labels[i]))
            error_count += 1.00
    print("错误率：%f%%" % (error_count / float(num_test_vecs) * 100))


def classify_person():
    result_list = ["not at all", "in small doses", "in large doses"]
    percent_tats = float(input("percentage of time spent playing video games?"))
    ff_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix("datingTestSet.txt")
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print(classifier_result)
    print("You will probably like this person:%s" % (result_list[classifier_result - 1]))


if __name__ == '__main__':
    # # 创建数据集
    # g, ls = create_data_set()
    # # 测试集
    # test = [0, 0]
    # # kNN分类
    # test_class = classify0(test, g, ls, 3)
    # # 打印分类结果
    # print(test_class)
    #
    # file = "datingTestSet.txt"
    # datingDataMat, datingLabels = file2matrix(file)
    # auto_norm(datingDataMat)
    dating_class_test()
    # classify_person()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15 * array(datingLabels), 15 * array(datingLabels))
    # plt.show()
