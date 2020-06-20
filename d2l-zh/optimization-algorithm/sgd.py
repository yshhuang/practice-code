"""
@Author: 	yshhuang@foxmail.com
@Date: 2020-06-10 17:44:14
@LastEditors: 	yshhuang@foxmail.com
@LastEditTime: 2020-06-11 18:12:35
@FilePath: /d2l-zh/optimization-algorithm/sgd.py
"""
import time
import numpy as np
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, data as gdata, loss as gloss


def get_data_ch7():
    data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data-data.mean(axis=0))/data.std(axis=0)
    return nd.array(data[:1500, :-1]), nd.array(data[:1500, -1])


def sgd(params, state, hyperparams):
    """
    @description:小批量随机梯度下降 
    @param {type} 
    @return: 
    """
    for p in params:
        p[:] -= hyperparams['lr']*p.grad


def train_ch7(trainer_fn, state, hyperparams, features, labels, batch_size=10, num_epochs=2):
    """
    @description:通用训练函数 
    @param {type} 
    @return: 
    """
    net, loss = d2l.linreg, d2l.squared_loss
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()

    def eval_loss():
        return loss(net(features, w, b), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(gdata.ArrayDataset(
        features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y).mean()
            l.backward()
            trainer_fn([w, b], state, hyperparams)
            if (batch_i+1)*batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss:%f,%f sec per epoch' % (ls[-1], time.time()-start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()


def train_sgd(lr, batch_size, num_epochs=2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)


if __name__ == "__main__":
    features, labels = get_data_ch7()
    print(features.shape)
    train_sgd(1, 1500, 6)
    train_sgd(0.005, 1)
    train_sgd(0.05, 10)

