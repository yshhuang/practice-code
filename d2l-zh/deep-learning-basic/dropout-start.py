# encoding:utf-8
"""
@Time    : 2020-05-14 13:38
@Author  : yshhuang@foxmail.com
@File    : dropout-start.py
@Software: PyCharm
"""

import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn


def dropout(X, drop_prob):
    """
    丢弃函数
    :param X:
    :param drop_prob:
    :return:
    """
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob
    return mask * X / keep_prob


def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, W1) + b1).relu()
    if autograd.is_training():
        H1 = dropout(H1, drop_prob1)
    H2 = (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():
        H2 = dropout(H2, drop_prob2)
    return nd.dot(H2, W3) + b3


if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
    b1 = nd.zeros(num_hiddens1)
    W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
    b2 = nd.zeros(num_hiddens2)
    W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
    b3 = nd.zeros(num_outputs)
    params = [W1, b1, W2, b2, W3, b3]
    for param in params:
        param.attach_grad()

    drop_prob1, drop_prob2 = 0.2, 0.5

    num_epochs, lr, batch_size = 5, 0.5, 256
    loss = gloss.SoftmaxCrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
