"""
@Author: 	yshhuang@foxmail.com
@Date: 2020-06-08 18:05:53
@LastEditors: 	yshhuang@foxmail.com
@LastEditTime: 2020-06-10 14:20:52
@FilePath: /d2l-zh/optimization-algorithm/optimization-and-deepling-learning.py
"""
import d2lzh as d2l
from mpl_toolkits import mplot3d
import numpy as np


def f(x):
    return x * np.cos(np.pi * x)


if __name__ == "__main__":
    d2l.set_figsize((4.5, 2.5))
    x = np.arange(-1.0, 2.0, 0.1)
    fig, = d2l.plt.plot(x, f(x))
    fig.axes.annotate('local minimum', xy=(-0.3, -0.25),
                      xytext=(-0.77, -1.0), arrowprops=dict(arrowstyle='->'))
    fig.axes.annotate('global minimum', xy=(1.1, -0.95),
                      xytext=(0.6, 0.8), arrowprops=dict(arrowstyle='->'))
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')
    d2l.plt.show()
