# encoding:utf-8
"""
@Time    : 2020-05-21 14:50
@Author  : yshhuang@foxmail.com
@File    : ResNet.py
@Software: PyCharm
"""
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import nn


class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False,
            strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)


def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_channels):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk


if __name__ == '__main__':
    # blk = Residual(3)
    # blk.initialize()
    # X = nd.random.uniform(shape=(4, 3, 6, 6))
    # print(blk(X).shape)
    # blk = Residual(6, use_1x1conv=True, strides=2)
    # blk.initialize()
    # print(blk(X).shape)
    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(), nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
    X = nd.random.uniform(shape=(1, 1, 224, 224))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)
    lr, num_epochs, batch_size, ctx = 0.05, 5, 256, d2l.try_gpu()
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer,
                  ctx, num_epochs)

