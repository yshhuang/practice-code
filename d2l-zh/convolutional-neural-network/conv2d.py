# encoding:utf-8
"""
@Time    : 2020-05-18 15:39
@Author  : yshhuang@foxmail.com
@File    : conv2d.py
@Software: PyCharm
"""
from mxnet import autograd, nd
from mxnet.gluon import nn


def corr2d(X, K):
    """
    二维互相关预算
    :param X: 输入数组
    :param k: 核数组
    :return:
    """
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


if __name__ == '__main__':
    X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    K = nd.array([[0, 1], [2, 3]])
    print(corr2d(X, K))

    X = nd.ones((6, 8))
    X[:, 2:6] = 0
    print(X)

    K = nd.array([[1, -1]])
    Y = corr2d(X, K)
    print(Y)

    conv2d = nn.Conv2D(1, kernel_size=(1, 2))
    conv2d.initialize()
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    for i in range(10):
        with autograd.record():
            Y_hat = conv2d(X)
            l = (Y_hat - Y) ** 2
        l.backward()
        conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
        if (i + 1) % 2 == 0:
            print('batch %d,loss %.3f' % (i + 1, l.sum().asscalar()))
    print(conv2d.weight.data().reshape((1, 2)))
