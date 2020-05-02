# encoding:utf-8
"""
@Time    : 2020-05-01 15:30
@Author  : yshhuang@foxmail.com
@File    : softmax-start.py
@Software: PyCharm
"""
import d2lzh as d2l
from mxnet import autograd, nd
import fashion_MNIST as fM


def softmax(X):
    """
    定义softmax函数
    :param X:
    :return:
    """
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition


def net(X):
    """
    定义softmax回归模型
    :param X:
    :return:
    """
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    """
    交叉熵损失函数
    :param y_hat:
    :param y:
    :return:
    """
    return -nd.pick(y_hat, y).log()


def accuracy(y_hat, y):
    """
    定义准确率函数
    :param y_hat:
    :param y:
    :return:
    """
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    """
    计算模型的准确率
    :param data_iter:
    :param net:
    :return:
    """
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
        params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f' %
              (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


if __name__ == '__main__':
    # 1.读取数据
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs = 784
    num_outputs = 10
    # 2.初始化模型参数
    W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
    b = nd.zeros(num_outputs)
    W.attach_grad()
    b.attach_grad()
    # 3.训练模型
    num_epochs, lr = 5, 0.1
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,
              batch_size, [W, b], lr)
    # 4.预测
    for X, y in test_iter:
        break
    true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    fM.show_fashion_mnist(X[0:9], titles[0:9])
