# -*- coding: UTF-8 -*-
from layer import Layer
import numpy as np
import Filter
from activators import ReluActivator,IdentityActivator
from func import *


class ConvLayer(Layer):
    def __init__(self, sizes, filters, c3=[]):
        Layer.__init__(self, sizes)
        if len(c3)>0 :
            self.c3 = c3
            self.c3type = True
        else:
            self.c3type = False

        # channel_number 是 每个人filter的深度 而不是 filter的数量
        self.output_height = np.shape(self.output_array)[1]
        self.output_width = np.shape(self.output_array)[2]
        self.zero_padding = 0


        self.filters = []
        self.bias = []
        self.activator = IdentityActivator()

        for filter in filters:
            self.filters.append(Filter.Filter(filter))
        self.filters = np.array(self.filters)
        self.channel_number = self.get_channel_number()
        self.filter_number = len(self.filters)
        self.filter_height = np.shape(self.filters[0].weights)[1]
        self.filter_width = np.shape(self.filters[0].weights)[2]
        self.zero_padding = 0
        self.stride = 1

    def forward(self, input_array):
        """
        计算卷积层的输出
        输出结果保存在self.output_array
        """
        self.input_array = input_array
        if self.input_array.ndim == 2:
            self.input_array = self.input_array.reshape((1,np.shape(self.input_array)[0],np.shape(self.input_array)[1]))

        self.padded_input_array = padding(self.input_array,
                                          self.zero_padding)
        self.input_width = np.shape(self.input_array)[2]
        self.input_height = np.shape(self.input_array)[1]
        print '-------------start convlayer forward---'
        print np.shape(input_array)
        for f in range(len(self.filters)):
            filter = self.filters[f]
            conv(
                self.padded_input_array[self.c3[f]] if self.c3type else self.padded_input_array,
                # self.padded_input_array,
                filter.get_weights(),
                self.output_array[f],
                self.stride,
                filter.get_bias())
        element_wise_op(self.output_array,
                            self.activator.forward)
            # c3层卷积结果


    def backward(self, input_array, sensitivity_array,
                 activator):
        """
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        """
        # self.forward(input_array)
        print '-------------start convlayer backward ---'
        self.bp_sensitivity_map(sensitivity_array,
                                self.activator)
        self.bp_gradient(sensitivity_array)

    def bp_sensitivity_map(self, sensitivity_array,
                           activator):
        '''
        计算传递到上一层的sensitivity map
        sensitivity_array: 本层的sensitivity map
        activator: 上一层的激活函数
        '''
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        # full卷积，对sensitivitiy map进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差
        # 但这个残差不需要继续向上传递，因此就不计算了
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width +
              self.filter_width - 1 - expanded_width) / 2
        padded_array = padding(expanded_array, zp)
        # 初始化delta_array，用于保存传递到上一层的
        # sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的
        # sensitivity map相当于所有的filter的
        # sensitivity map之和
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重翻转180度
            flipped_weights = np.array(map(
                lambda i: np.rot90(i, 2),
                filter.get_weights()))
            # 计算与一个filter对应的delta_array

            delta_array = self.create_delta_array(np.shape(flipped_weights)[0] if self.c3type else 0)
            cur = f
            if self.c3type:
                pass
                # cur = self.c3[f]
                # print '------ c3 特殊 ------'
                # print '被卷积的 &(误差项) 部分：',np.shape(padded_array[cur])
                # print np.shape(flipped_weights)
                # print '匹配的delta 深度: ',np.shape(flipped_weights)[0]
                # print 'delta_array:',np.shape(delta_array)
            for d in range(delta_array.shape[0]):
                # 从 0 开始到 filter 的个数次
                # 将 此filter 每一层卷积 第 f 个 sensitivity_array
                # 最后求和
                conv(padded_array[cur], flipped_weights[d],
                     delta_array[d], 1, 0)
            # self.delta_array += delta_array
            self.deltaCalculate(delta_array)
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_array):
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        for f in range(self.filter_number):
            # 计算每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d],
                     expanded_array[f],
                     filter.weights_grad[d], 1, 0)
            # 计算偏置项的梯度
            filter.bias_grad = expanded_array[f].sum()
    def expand_sensitivity_map(self, sensitivity_array):
        if sensitivity_array.ndim == 2:
            sensitivity_array = sensitivity_array.reshape(np.shape(sensitivity_array)[0], np.shape(sensitivity_array)[1], 1)
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expanded_width = (self.input_width -
                          self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height -
                           self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height,
                                 expanded_width))
        # 从原始sensitivity map拷贝误差值8
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = \
                    sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self, num = 0):
        if num == 0:
            num = self.channel_number
        return np.zeros((num, self.input_height, self.input_width))
    def get_channel_number(self):
        vmax = 0
        for f in self.filters:
            if np.shape(f.get_weights())[0]>=vmax:
                vmax = np.shape(f.get_weights())[0]
        return vmax
    def deltaCalculate(self, delta):
        # 对c3 层做的特殊处理，由于 偏差项 为6，14，14 而卷集合的尺寸 有 （3，14，14）（4，14，14）（6，14，14）
        # 由于尺寸不同不能直接相加 我将 偏差项的 相应部分相加， 达到目标
        self.delta_array[:np.shape(delta)[0]] +=delta