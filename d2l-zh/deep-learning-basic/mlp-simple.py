# encoding:utf-8
"""
@Time    : 2020-05-02 22:42
@Author  : yshhuang@foxmail.com
@File    : mlp-simple.py
@Software: PyCharm
"""
import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

if __name__ == '__main__':
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'), nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
    num_epochs = 5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
                  batch_size, None, None, trainer)
