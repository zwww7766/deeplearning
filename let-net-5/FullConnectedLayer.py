# -*- coding: utf-8 -*-
# 全连接层实现类
import  numpy as np
from activators import SigmoidActivator
import copy
class FullConnectedLayer(object):
        def __init__(self, output_size, input_size):
            """
            构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
            """
            self.input_size = input_size
            self.output_size = output_size
            self.activator = SigmoidActivator()
            # 权重数组W
            Fi = input_size+1
            self.W = np.random.uniform(-2.4/Fi, 2.4/Fi,
                                       (output_size, input_size))
            # 偏置项b-
            self.b = np.zeros((output_size, 1))
            # 输出向量
            self.output_array = np.zeros((output_size, 1))

        def forward(self, input_array):
            """前向计算
            input_array: 输入向量，维度必须等于input_size
            """
            #  式2
            self.input_array = input_array.reshape([input_array.shape[0] * input_array.shape[1] * input_array.shape[2]])
            self.input_array = np.array(input_array).reshape(len(input_array), 1)
            self.output_array = self.activator.forward(
                np.dot(self.W, self.input_array) + self.b)

        def backward(self, input_array, delta_array, learn_rate):
            """反向计算W和b的梯度
            delta_array: 从上一层传递过来的误差项
            """
            # 式8
            self.delta_array = None
            delta_array = delta_array.reshape([84,1])
            errm = self.output_array * delta_array
            self.delta_array =np.dot( errm.reshape((84,)), self.W).reshape((120,1))


            self.W_grad = np.dot(errm, self.input_array.T)
            self.b_grad = delta_array
            self.update(learn_rate)


        def update(self, learning_rate):
            '''
            使用梯度下降算法更新权重
            '''
            self.W += learning_rate * self.W_grad
            self.b += learning_rate * self.b_grad