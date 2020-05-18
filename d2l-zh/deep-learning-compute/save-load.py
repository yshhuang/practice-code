# encoding:utf-8
"""
@Time    : 2020-05-18 14:30
@Author  : yshhuang@foxmail.com
@File    : save-load.py
@Software: PyCharm
"""
from mxnet import nd
from mxnet.gluon import nn


class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))


if __name__ == '__main__':
    x = nd.ones(3)
    nd.save('x', x)
    x2 = nd.load('x')
    print(x2)

    y = nd.zeros(4)
    nd.save('xy', [x, y])
    x2, y2 = nd.load('xy')
    print(x2, y2)

    mydict = {'x': x, 'y': y}
    nd.save('mydict', mydict)
    mydict2 = nd.load('mydict')
    print(mydict2)

    net = MLP()
    net.initialize()
    X = nd.random.uniform(shape=(2, 20))
    Y = net(X)
    filename = 'mlp.params'
    net.save_parameters(filename)

    net2 = MLP()
    net2.load_parameters(filename)
    Y2 = net2(X)
    print(Y2 == Y)
