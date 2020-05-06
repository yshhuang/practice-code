# encoding:utf-8
"""
@Time    : 2020-04-27 16:06
@Author  : yshhuang@foxmail.com
@File    : linear-regression-simple.py
@Software: PyCharm
"""
from mxnet import autograd, nd, init, gluon
from mxnet.gluon import data as gdata, nn, loss as gloss

if __name__ == '__main__':
    # 1.生成数据集
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    # 2.读取数据集
    batch_size = 10
    dataset = gdata.ArrayDataset(features, labels)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    # 3.定义模型
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
    # 4.训练模型
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), labels)
        print('epoch %d,loss: %f' % (epoch, l.mean().asnumpy()))

    dense = net[0]
    print(true_w, dense.weight.data())
    print(true_b, dense.bias.data())
