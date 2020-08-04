"""
@Author: 	yshhuang@foxmail.com
@Date: 2020-07-23 16:13:08
@LastEditors: 	yshhuang@foxmail.com
@LastEditTime: 2020-07-27 17:47:49
@FilePath: /d2l-zh/srcnn/model.py
"""

from mxnet.gluon import nn
from mxnet import initializer


class SrCnn(nn.Sequential):
    def __init__(self, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False, force_reinit=False):
        self.add(nn.Conv2D(kernel_size=9,
                           channels=64, activation='relu'))
        self.add(nn.Conv2D(kernel_size=1,
                           channels=32, activation='relu'))
        self.add(nn.Conv2D(kernel_size=5, channels=3))
        return super().initialize(init=init, ctx=ctx, verbose=verbose, force_reinit=force_reinit)
