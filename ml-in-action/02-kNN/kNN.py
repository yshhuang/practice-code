from numpy import *
import operator


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 1.0]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)
    distance = sq_distance ** 0.5
    sorted_dist_indices = distance.argsort()
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    # 创建数据集
    g, ls = create_data_set()
    # 测试集
    test = [0, 0]
    # kNN分类
    test_class = classify0(test, g, ls, 3)
    # 打印分类结果
    print(test_class)
