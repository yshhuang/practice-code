# encoding:utf-8
"""
@Time    : 2020-05-06 13:45
@Author  : yshhuang@foxmail.com
@File    : weight-damping-start.py
@Software: PyCharm
"""

import d2lzh as d2l
from mxnet import autograd, nd
from mxnet.gluon import data as gdata


def init_params():
    """
    初始化模型参数
    :return:
    """
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]


def l2_penalty(w):
    """
    l2范数惩罚项
    :param w:
    :return:
    """
    return (w ** 2).sum() / 2


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b),
                            test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('l2 norm of w:', w.norm().asscalar())


if __name__ == '__main__':
    n_train, n_test, num_inputs = 20, 100, 200
    true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05
    features = nd.random.normal(shape=(n_train + n_test, num_inputs))
    labels = nd.dot(features, true_w) + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    train_features, test_features = features[:n_train, :], features[n_train:, :]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    batch_size, num_epochs, lr = 1, 100, 0.003
    net, loss = d2l.linreg, d2l.squared_loss
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)

    fit_and_plot(lambd=3)
