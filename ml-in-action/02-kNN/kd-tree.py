# -*- coding: UTF-8 -*-
"""
@author: yshhuang@foxmail.com
@file: .py
@time: 2018/8/21 5:37 PM
"""

import numpy as np
import heapq
import itertools
import operator
import math
from collections import deque
from functools import wraps


class KDNode(object):
    def __init__(self, data, axis, left=None, right=None):
        self.data = data
        self.axis = axis
        self.left = left
        self.right = right

    def __repr__(self):
        return np.array2string(self.data)

    def __str__(self, level=0):
        ret = repr(self.data) + "\n"
        if self.left is not None:
            ret += "\t" * (level + 1) + "left:\t" + self.left.__str__(level + 1)
        if self.right is not None:
            ret += "\t" * (level + 1) + "right:\t" + self.right.__str__(level + 1)
        return ret


def create_kd_tree(data_mat):
    length = len(data_mat)
    if length == 0:
        return
    var = data_mat.var(axis=0)
    split = np.where(var == var.max())[0][0]
    data_mat = data_mat[data_mat[:, split].argsort()]  # 将特征矩阵按分割轴进行排序
    data = data_mat[int(length / 2)]
    root = KDNode(data, split)
    root.left = create_kd_tree(data_mat[0:int(length / 2)])
    root.right = create_kd_tree(data_mat[int(length / 2) + 1:length])
    return root


def computer_distance(a, b):
    d = ((a - b) ** 2).sum() ** 0.5
    return d


def find_max_distance(data_set, target):
    max_i = 0
    max_distance = 0
    for i in range(len(data_set)):
        distance = computer_distance(data_set[i].data, target)
        if distance > max_distance:
            max_i = i
            max_distance = distance
    return max_i, distance


def find_knn(root, query, k):
    result = []
    search_path = []  # 存储搜索路径上的节点

    while root is not None:
        search_path.append(root)
        axis = root.axis
        if root.left is None and root.right is None:
            result.append(root)
        if query[axis] <= root.data[axis]:
            root = root.left
        else:
            root = root.right

    while search_path:
        back_point = search_path.pop()
        print(back_point.data)
        print(result)
        print("------")
        axis = back_point.axis
        max_i, max_distance = find_max_distance(result, query)
        if len(result) < k or abs(query[axis] - back_point.data[axis]) < max_distance:
            if query[axis] <= back_point.data[axis]:
                root = back_point.right
            else:
                root = back_point.left
            if root is not None:
                search_path.append(root)
                distance = computer_distance(root.data, query)
                if len(result) == k and distance < max_distance:
                    result[max_i] = back_point
                elif len(result) < k:
                    result.append(back_point)
    return result


if __name__ == '__main__':
    mat = np.array([[1, 8, 3], [3, 4, 7], [3, 6, 9], [3, 6, 8]])
    tree = create_kd_tree(mat)
    r = find_knn(tree, [1, 8, 2], 4)
    print(r)
    # mat = sort_by_dist(mat, [3, 6, 8])
    # print(mat)
