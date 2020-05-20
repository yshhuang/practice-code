# encoding:utf-8
"""
@Time    : 2020-05-20 18:00
@Author  : yshhuang@foxmail.com
@File    : AlexNet.py
@Software: PyCharm
"""

import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, nn
import os
import sys


def load_data_fashion_mnist(batch_size, resize=None,
        root=os.path.join('~', '.mxnet', 'databases', 'fashion-minst')):
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size,
                                  shuffle=True, num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size,
                                 shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


if __name__ == '__main__':
    net = nn.Sequential()
    net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    X = nd.random.uniform(shape=(1, 1, 224, 224))
    # net.initialize()
    # for layer in net:
    #     X = layer(X)
    #     print(layer.name, 'output shape:\t', X.shape)

    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs, ctx = 0.01, 5, d2l.try_gpu()
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
