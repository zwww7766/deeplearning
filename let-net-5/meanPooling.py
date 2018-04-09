# -*- coding: UTF-8 -*-
from layer import Layer
import numpy as np
import Filter
from func import *

class meanPoolingLayer(Layer):
    def __init__(self, sizes, filters):
        Layer.__init__(self, sizes)

        self.channel_number = np.shape(self.output_array)[0]
        self.output_height = np.shape(self.output_array)[1]
        self.output_width = np.shape(self.output_array)[2]

        self.filters = []
        self.bias = []

        for filter in filters:
            self.filters.append(Filter.Filter(filter))
        self.filters = np.array(self.filters)
        # print len(self.filters)
        # print np.shape(self.filters[0].weights)
        self.filter_number = np.shape(len(self.filters))
        self.filter_height = np.shape(self.filters[0].weights)[0]
        self.filter_width = np.shape(self.filters[0].weights)[1]

        self.zero_padding = 1
        self.stride = 1


    def forward(self, input_array):
        self.input_array = input_array
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_average(
                            get_patch(input_array[d], i, j,
                                  self.filter_width,
                                  self.filter_height,
                                  self.stride)))

    def backward(self, input_array, sensitivity_array, t = []):
        self.delta_array = np.zeros(self.input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        self.input_array[d], i, j,
                        self.filter_width,
                        self.filter_height,
                        self.stride)
                    for m in range(patch_array.shape[0]):
                        for n in range(patch_array.shape[1]):
                            self.delta_array[d,
                                             i * self.stride + m,
                                             j * self.stride + n] = \
                                sensitivity_array[d, i, j]/4
