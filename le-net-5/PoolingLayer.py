# -*- coding: UTF-8 -*-
import layer
import numpy as np
from func import *

"""
下采样 pooling layer
四个输入相加乘以可训练参数poolparas[0] 再加上一个偏置项poolparas[1] 结果通过sigmoid
此处为 Max pooling
"""


class PoolingLayer(layer):

#     def __init__(self, input_width, input_height,
#                  channel_number, filter_width,
#                  filter_height, stride):
#         self.input_width = input_width
#         self.input_height = input_height
#         self.channel_number = channel_number
#         self.filter_width = filter_width
#         self.filter_height = filter_height
#         self.stride = stride
#         self.output_width = (input_width -
#                              filter_width) / self.stride + 1
#         self.output_height = (input_height -
#                               filter_height) / self.stride + 1
#         self.output_array = np.zeros((self.channel_number,
#                                       self.output_height, self.output_width))

    def __init__(self, lay_size=[], filters=[]):
        """
        maxpooling 层 没有可训练参数
        :param lay_size:
        :param filters:
        """
        layer.__init__(self, lay_size)
        self.channel_number = np.shape(self.output_array)[0]
        self.output_height = np.shape(self.output_array)[1]
        self.output_width = np.shape(self.output_array)[2]
        self.stride = 1
        self.filter_height = 2
        self.filter_width = 2

    def forward(self, input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_patch(input_array[d], i, j,
                                  self.filter_width,
                                  self.filter_height,
                                  self.stride).max())

    def backward(self, input_array, sensitivity_array, activator = None):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        input_array[d], i, j,
                        self.filter_width,
                        self.filter_height,
                        self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d,
                                     i * self.stride + k,
                                     j * self.stride + l] = \
                        sensitivity_array[d, i, j]
