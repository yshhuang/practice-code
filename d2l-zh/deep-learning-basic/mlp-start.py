# encoding:utf-8
"""
@Time    : 2020-05-02 22:27
@Author  : yshhuang@foxmail.com
@File    : mlp-start.py
@Software: PyCharm
"""
import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss


def relu(X):
    """
    定义激活函数
    :param X:
    :return:
    """
    return nd.maximum(X, 0)


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2


if __name__ == '__main__':
    # 1.定义模型参数
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
    b1 = nd.zeros(num_hiddens)
    W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
    b2 = nd.zeros(num_outputs)
    params = [W1, b2, W2, b2]
    for param in params:
        param.attach_grad()
    loss = gloss.SoftmaxCrossEntropyLoss()
    # 2.训练模型
    num_epochs, lr = 5, 0.5
    d2l.train_ch3(net, test_iter, test_iter, loss, num_epochs, batch_size,
                  params, lr)
