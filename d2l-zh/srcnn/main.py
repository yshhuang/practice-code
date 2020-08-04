"""
@Author: 	yshhuang@foxmail.com
@Date: 2020-07-24 11:02:58
@LastEditors: 	yshhuang@foxmail.com
@LastEditTime: 2020-07-24 17:00:12
@FilePath: /d2l-zh/srcnn/main.py
"""

from model import SrCnn
from model import (try_gpu)
from mxnet import autograd, gluon, init, nd

if __name__ == "__main__":
    srcnn = SrCnn()
    srcnn.initialize(ctx=try_gpu(), force_reinit=True, init=init.Xavier())
    srcnn.train()
