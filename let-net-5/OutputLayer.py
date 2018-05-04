# -*- coding: utf-8 -*-
import numpy as np

"""
rbf 径向基函数
欧式距离
"""
class OutputLayer(object):
    def __init__(self, output_size, input_size):
        self.input_size = input_size
        self.output_size = output_size
        # 输出向量
        self.output_array = np.zeros((output_size,1))
        self.weight = np.float64(np.random.choice([-1,1], [output_size, input_size])) #10,84

    def forward(self, input_array):
        self.input_array = input_array.reshape([input_array.shape[0] * input_array.shape[1]])
        output = np.matmul(self.weight,self.input_array)
        max_ineach = max(output)
        output = np.exp(output - max_ineach)
        output = output/sum(output)
        self.output_array = output.reshape([1,1,10])

    def backward(self, input_array, sensitivity_array, learn_rate):
        self.delta_array = None
        input_array = input_array.reshape([1,1,84])
        current_error_matrix = np.array(np.matrix(list(sensitivity_array[0]) * self.weight.shape[1]).T)
        weight_update = (self.weight - np.array(list(input_array[0]) * self.weight.shape[0])) * current_error_matrix
        for i in range(10):
            weight_update[i] = (sensitivity_array[0][0][i] - self.output_array[0][0][i]) * self.input_array
        self.weight -= learn_rate * weight_update
        self.delta_array = ((np.array(list(input_array[0]) * self.weight.shape[0]) - self.weight) * current_error_matrix).sum(axis = 0)


