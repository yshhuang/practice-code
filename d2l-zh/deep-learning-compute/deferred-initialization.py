# encoding:utf-8
"""
@Time    : 2020-05-14 20:57
@Author  : yshhuang@foxmail.com
@File    : deferred-initialization.py
@Software: PyCharm
"""
from mxnet import init, nd
from mxnet.gluon import nn


class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)


if __name__ == '__main__':
    # net = nn.Sequential()
    # net.add(nn.Dense(256, activation='relu'), nn.Dense(10))
    # net.initialize(init=MyInit())
    # X = nd.random.uniform(shape=(2, 20))
    # Y = net(X)
    # net.initialize(init=MyInit(), force_reinit=True)

    net = nn.Sequential()
    net.add(nn.Dense(256, in_units=20, activation='relu'))
    net.add(nn.Dense(10, in_units=256))
    net.initialize(init=MyInit())
