"""
@Author: 	yshhuang@foxmail.com
@Date: 2020-06-10 14:22:06
@LastEditors: 	yshhuang@foxmail.com
@LastEditTime: 2020-06-10 14:22:07
@FilePath: /d2l-zh/deep-learning-basic/fashion_MNIST.py
"""
# encoding:utf-8
"""
@Time    : 2020-04-27 18:45
@Author  : yshhuang@foxmail.com
@File    : fashion_MNIST.py
@Software: PyCharm
"""
import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time


def get_fashion_mnist_labels(labels):
    """
    将数值标签转成相应的文本标签
    :param labels:
    :return:
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    d2l.plt.show()


if __name__ == '__main__':
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)
    print(len(mnist_train), len(mnist_test))

    X, y = mnist_train[0:9]
    show_fashion_mnist(X, get_fashion_mnist_labels(y))
    batch_size = 256
    transformer = gdata.vision.transforms.ToTensor
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True, num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False, num_workers=num_workers)
