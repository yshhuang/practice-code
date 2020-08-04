"""
@Author: 	yshhuang@foxmail.com
@Date: 2020-07-27 16:57:35
@LastEditors: 	yshhuang@foxmail.com
@LastEditTime: 2020-07-30 20:05:42
@FilePath: /d2l-zh/srcnn/train.py
"""

from preprocessing import (generate_data, try_gpu, data_iter)
import os
import h5py
from mxnet import nd, gluon, autograd
from model import SrCnn
from mxnet.gluon import loss as gloss
import time
import random

train_data = '../data/srcnn/Train/'
lr = 1e-4
epoch = 10
batch_size = 128

if __name__ == "__main__":
    if not os.path.exists("train.h5"):
        generate_data(train_data, "train.h5")
    with h5py.File("train.h5", 'r') as hf:
        train_input = nd.array(hf.get('input'))
        train_label = nd.array(hf.get('label'))
    net = SrCnn()
    net.initialize(ctx=try_gpu())
    if os.path.exists("srcnn.params"):
        net.load_parameters("srcnn.params")
    ctx = try_gpu()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    print('training on', ctx)
    loss = gloss.L2Loss()
    for ep in range(epoch):
        train_l_sum,  n, start = 0.0, 0, time.time()
        # batch_idxs = len(train_input) // batch_size

        for X, y in data_iter(batch_size, train_input, train_label):
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            X = nd.transpose(X, (0, 3, 1, 2))
            y = nd.transpose(y, (0, 3, 1, 2))
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            print(y.size)
            n += y.size
        print('epoch %d,loss %f' % (ep+1, train_l_sum/n))
        net.save_parameters("srcnn.params")
