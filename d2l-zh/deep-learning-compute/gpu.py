# encoding:utf-8
"""
@Time    : 2020-05-18 15:05
@Author  : yshhuang@foxmail.com
@File    : gpu.py
@Software: PyCharm
"""
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

if __name__ == '__main__':
    print(mx.cpu(), mx.gpu(), mx.gpu(1))
    x = nd.array([1, 2, 3],ctx=mx.gpu())
    print(x.context)
