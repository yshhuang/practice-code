# encoding:utf-8
"""
@Time    : 2020-04-27 14:43
@Author  : yshhuang@foxmail.com
@File    : linear-regression-start.py
@Software: PyCharm
"""

from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random


def use_svg_display():
    """
    用矢量图显示
    :return:
    """
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """
    设置图的尺寸
    :param figsize:
    :return:
    """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    """
    读取数据集
    :param batch_size:
    :param features:
    :param labels:
    :return:
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)


def linreg(X, w, b):
    """
    线性回归模型的定义
    :param X:
    :param w:
    :param b:
    :return:
    """
    return nd.dot(X, w) + b


def squared_loss(y_hat, y):
    """
    定义损失函数
    :param y_hat:
    :param y:
    :return:
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    定义优化算法
    :param params:
    :param lr:
    :param batch_size:
    :return:
    """
    for param in params:
        param[:] = param - lr * param.grad / batch_size


if __name__ == '__main__':
    # 生成数据集 y = Xw + b + ε,
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)

    set_figsize()
    plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
    # plt.show()
    # 读取数据集
    batch_size = 10
    # for X, y in data_iter(batch_size, features, labels):
    #     print(X, y)
    #     break
    # 初始化模型参数
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    # 训练模型
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            with autograd.record():
                l = loss(net(X, w, b), y)
            l.backward()
            sgd([w, b], lr, batch_size)
        train_l = loss(net(features, w, b), labels)
        print('epoch %d,loss %f' % (epoch + 1, train_l.mean().asnumpy()))

    print(true_w, w)
    print(true_b, b)
