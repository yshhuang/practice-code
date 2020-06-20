"""
@Author: 	yshhuang@foxmail.com
@Date: 2020-06-11 20:39:03
@LastEditors: 	yshhuang@foxmail.com
@LastEditTime: 2020-06-11 20:49:20
@FilePath: /d2l-zh/optimization-algorithm/momentum.py
"""
import d2lzh as d2l
from mxnet import nd


def init_momentum_states():
    v_w = nd.zeros((features.shape[1], 1))
    v_b = nd.zeros(1)
    return (v_w, v_b)


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum']*v+hyperparams['lr']*p.grad
        p[:] -= v


if __name__ == "__main__":
    eta = 0.4
    features, labels = d2l.get_data_ch7()
    d2l.train_ch7(sgd_momentum, init_momentum_states(), {
                  'lr': 0.02, 'momentum': 0.5}, features, labels)
    d2l.train_gluon_ch7(
        'sgd', {'learning_rate': 0.004, 'momentum': 0.9}, features, labels)
