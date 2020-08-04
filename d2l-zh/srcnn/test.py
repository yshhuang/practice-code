"""
@Author: 	yshhuang@foxmail.com
@Date: 2020-07-27 17:45:59
@LastEditors: 	yshhuang@foxmail.com
@LastEditTime: 2020-07-30 17:28:00
@FilePath: /d2l-zh/srcnn/test.py
"""

from preprocessing import (try_gpu, modcrop)
from model import SrCnn
import os
import cv2 as cv
import h5py
from mxnet import nd
import numpy as np

test_data = '../data/srcnn/Test/Set5/'
test_image = '/home/deploy/waifu2x/test/valid_lr.png'
input_size = 33
label_size = 21
scale = 3
stride = 14


def propressing(image, output):
    sub_inputs = []
    sub_labels = []
    padding = int(abs(input_size-label_size)/2)

    mat = cv.imread(image, cv.IMREAD_COLOR)
    mat = mat/255
    label_ = modcrop(mat, scale)
    input_ = cv.resize(label_, (0, 0), fx=1./scale, fy=1. /
                       scale, interpolation=cv.INTER_AREA)
    input_ = cv.resize(input_, (0, 0), fx=scale,
                       fy=scale, interpolation=cv.INTER_CUBIC)
    h, w, _ = input_.shape
    nx = ny = 0
    for x in range(0, h-input_size+1, stride):
        nx += 1
        ny = 0
        for y in range(0, w-input_size+1, stride):
            ny += 1
            sub_input = input_[x:x+input_size, y:y+input_size]
            sub_label = label_[x+padding:x+padding +
                               label_size, y+padding:y+padding+label_size]
            sub_inputs.append(sub_input)
            sub_labels.append(sub_label)
    with h5py.File(output, 'w') as f:
        arrinput = np.asarray(sub_inputs)
        f.create_dataset('input', data=arrinput)
        f.create_dataset('label', data=sub_labels)
    return nx, ny


def merge(images, size):
    h, w = images.shape[2], images.shape[3]
    img = np.zeros((h*size[0], w*size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        image = nd.transpose(image, (1, 2, 0)).asnumpy()
        img[j*h:j*h+h, i*w:i*w+w, 0:3] = image
    return img


if __name__ == "__main__":
    nx, ny = propressing(test_image, "test.h5")
    with h5py.File("test.h5", 'r') as hf:
        test_input = nd.array(hf.get('input'))
        test_label = nd.array(hf.get('label'))

    net = SrCnn()
    net.initialize(ctx=try_gpu())
    if os.path.exists("srcnn.params"):
        net.load_parameters("srcnn.params")
    ctx = try_gpu()
    test_input = test_input.as_in_context(ctx)
    test_input = nd.transpose(test_input, (0, 3, 1, 2))
    result = net(test_input)
    print(result[0, 0, 0, 0:10])
    result = merge(result, [nx, ny])
    print(result.shape)
    cv.imwrite("kk.png", result*255)
