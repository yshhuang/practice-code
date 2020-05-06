# encoding:utf-8
"""
@Time    : 2020-05-02 21:33
@Author  : yshhuang@foxmail.com
@File    : softmax-simple.py
@Software: PyCharm
"""
import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
import fashion_MNIST as fM

if __name__ == '__main__':
    # 1.初始化模型参数
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    net = nn.Sequential()
    net.add(nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))
    # 2.定义损失函数
    loss = gloss.SoftmaxCrossEntropyLoss()
    # 3.定义优化算法
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
    # 4.训练模型
    num_epochs = 5
    d2l.train_ch3(net, test_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)
    # 5.预测
    for X, y in test_iter:
        break
    true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    fM.show_fashion_mnist(X[0:9], titles[0:9])
