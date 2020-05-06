# encoding:utf-8
"""
@Time    : 2020-05-04 17:27
@Author  : yshhuang@foxmail.com
@File    : fitting-test.py
@Software: PyCharm
"""

import d2lzh as d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
        legend=None, figsize=(3.5, 2.5)):
    """
    作图函数
    :param x_vals:
    :param y_vals:
    :param x_label:
    :param y_label:
    :param x2_vals:
    :param y2_vals:
    :param legend:
    :param figsize:
    :return:
    """
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels),
                                  batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    print('final epoch:train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net[0].weight.data().asnumpy(),
          '\nblias', net[0].bias.data().asnumpy())


if __name__ == '__main__':
    n_train, n_test, true_w, true_b = 100, 100, [1.2, - 3.4, 5.6], 5
    features = nd.random.normal(shape=(n_train + n_test, 1))
    poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))
    labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
              + true_w[2] * poly_features[:, 2] + true_b)
    labels += nd.random.normal(scale=0.1, shape=labels.shape)

    num_epochs, loss = 100, gloss.L2Loss()
    # fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
    #              labels[:n_train], labels[n_train:])
    # fit_and_plot(features[:n_train, :], features[n_train:, :],
    #              labels[:n_train], labels[n_train:])
    fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2], labels[n_train:])
