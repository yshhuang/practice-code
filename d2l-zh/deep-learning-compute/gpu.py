# encoding:utf-8
"""
@Time    : 2020-05-18 15:05
@Author  : yshhuang@foxmail.com
@File    : gpu.py
@Software: PyCharm
"""
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

if __name__ == '__main__':
    print(mx.cpu(), mx.gpu())
    x = nd.array([1, 2, 3], ctx=mx.gpu())
    print(x.context)

    y = x.copyto(mx.gpu())
    print(y)

    z = x.as_in_context(mx.gpu())
    print(z)

    print(y.as_in_context(mx.gpu()) is y)
    print(y.copyto(mx.gpu()) is y)

    print((z + 2).exp() * y)

    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(ctx=mx.gpu())
    print(net(y))
    print(net[0].weight.data())
