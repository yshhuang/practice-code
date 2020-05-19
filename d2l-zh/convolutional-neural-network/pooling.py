# encoding:utf-8
"""
@Time    : 2020-05-18 20:19
@Author  : yshhuang@foxmail.com
@File    : pooling.py
@Software: PyCharm
"""
from mxnet import nd
from mxnet.gluon import nn


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[1]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y


if __name__ == '__main__':
    X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    print(pool2d(X, (2, 2)))
    print(pool2d(X, (2, 2), 'avg'))

    X = nd.arange(16).reshape((1, 1, 4, 4))
    pool2d = nn.MaxPool2D(3, padding=1, strides=2)
    print(pool2d(X))

    pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
    print(pool2d(X))

    X = nd.concat(X, X + 1, dim=1)
    pool2d = nn.MaxPool2D(3, padding=1, strides=2)
    print(pool2d(X))
