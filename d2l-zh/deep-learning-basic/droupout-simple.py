# encoding:utf-8
"""
@Time    : 2020-05-14 14:15
@Author  : yshhuang@foxmail.com
@File    : droupout-simple.py
@Software: PyCharm
"""

import d2lzh as d2l
from mxnet.gluon import nn, loss as gloss
from mxnet import nd, init, gluon

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
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'),
            nn.Dropout(drop_prob1),
            nn.Dense(256, activation='relu'),
            nn.Dropout(drop_prob2),
            nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))

    num_epochs, lr, batch_size = 5, 0.5, 256
    loss = gloss.SoftmaxCrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)
