# encoding:utf-8
"""
@Time    : 2020-05-14 18:59
@Author  : yshhuang@foxmail.com
@File    : model-param.py
@Software: PyCharm
"""
from mxnet import init, nd
from mxnet.gluon import nn


class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5


if __name__ == '__main__':
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    net.initialize()
    X = nd.random.uniform(shape=(2, 20))
    Y = net(X)
    net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
    print(net[0].weight.data()[0])
    net.initialize(init=init.Constant(1), force_reinit=True)
    print(net[0].weight.data()[0])
    net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
    print(net[0].weight.data()[0])
    net.initialize(MyInit(), force_reinit=True)
    print(net[0].weight.data()[0])
    net[0].weight.set_data(net[0].weight.data() + 1)
    print(net[0].weight.data()[0])
    # ===============
    net = nn.Sequential()
    shared = nn.Dense(8, activation='relu')
    net.add(nn.Dense(8, activation='relu'), shared,
            nn.Dense(8, activation='relu', params=shared.params),
            nn.Dense(10))
    net.initialize()
    net(X)
    print(net[1].weight.data()[0] == net[2].weight.data()[0])
