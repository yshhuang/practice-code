"""
@Author: 	yshhuang@foxmail.com
@Date: 2020-07-23 17:24:04
@LastEditors: 	yshhuang@foxmail.com
@LastEditTime: 2020-07-30 18:41:41
@FilePath: /d2l-zh/srcnn/preprocessing.py
"""
import numpy as np
import os
import cv2 as cv
import h5py
import mxnet
from mxnet import nd
import random


train_data = '../data/srcnn/Train/'
test_data = '../data/srcnn/Test/Set5/'
input_size = 33
label_size = 21
scale = 3
stride = 14


def try_gpu():
    try:
        ctx = mxnet.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mxnet.base.MXNetError:
        ctx = mxnet.cpu()
    return ctx


def modcrop(image, scale=3):
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h-h % scale
        w = w-w % scale
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h-h % scale
        w = w-w % scale
        image = image[0:h, 0:w]
    return image


def generate_data(folder, output):
    sub_inputs = []
    sub_labels = []
    padding = int(abs(input_size-label_size)/2)
    allImage = os.listdir(folder)
    for image in allImage:
        mat = cv.imread(folder+image, cv.IMREAD_COLOR)
        mat = mat/255
        label_ = modcrop(mat, scale)
        input_ = cv.resize(label_, (0, 0), fx=1./scale, fy=1./scale,
                           interpolation=cv.INTER_AREA)
        input_ = cv.resize(input_, (0, 0), fx=scale,
                           fy=scale, interpolation=cv.INTER_CUBIC)
        h, w, _ = input_.shape

        for x in range(0, h-input_size+1, stride):
            for y in range(0, w-input_size+1):
                sub_input = input_[x:x+input_size, y:y+input_size]
                sub_label = label_[x+padding:x+padding +
                                   label_size, y+padding:y+padding+label_size]
                sub_inputs.append(sub_input)
                sub_labels.append(sub_label)
    with h5py.File(output, 'w') as f:
        f.create_dataset('input', data=sub_inputs)
        f.create_dataset('label', data=sub_labels)


def data_iter(batch_size, features, labels):
    """
    读取数据集
    :param batch_size:
    :param features:
    :param labels:
    :return:
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)


if __name__ == "__main__":
    generate_data(train_data, 'train.h5')
    generate_data(test_data, 'test.h5')
