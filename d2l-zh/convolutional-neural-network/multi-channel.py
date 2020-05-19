# encoding:utf-8
"""
@Time    : 2020-05-18 18:35
@Author  : yshhuang@foxmail.com
@File    : multi-channel.py
@Software: PyCharm
"""
import d2lzh as d2l
from mxnet import nd


def corr2d_multi_in(X, K):
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])


def corr2d_multi_in_out(X, K):
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)
    return Y.reshape((c_o, h, w))


if __name__ == '__main__':
    X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
    print(corr2d_multi_in(X, K))

    K = nd.stack(K, K + 1, K + 2)
    print(K.shape)
    print(corr2d_multi_in_out(X, K))

    X = nd.random.uniform(shape=(3, 3, 3))
    K = nd.random.uniform(shape=(2, 3, 1, 1))

    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    print((Y1 - Y2).norm().asscalar() < 1e-6)
