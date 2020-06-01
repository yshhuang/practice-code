# encoding:utf-8
"""
@Time    : 2020-05-25 18:51
@Author  : yshhuang@foxmail.com
@File    : rnn-simple.py
@Software: PyCharm
"""
import d2lzh as d2l
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time

import  d2lzh as d2l
import math