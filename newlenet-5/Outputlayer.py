# -*- coding:utf8 -*-
from Layer import *
from FullconnectedLayer import *
"""
rbf 径向基函数
欧式距离
"""
class OutputLayer(FullconnectedLayer):
    def __init__(self, output_size, input_size):
        # 输出向量
        FullconnectedLayer.__init__(self, output_size, input_size)
        self.weight = np.float64(np.random.choice([-1, 1], [output_size, input_size]))  # 10,84

    def forward(self, input):
        self.input_array = input.reshape([input.shape[0] * input.shape[1] * input.shape[2]])
        output = np.matmul(self.weight, self.input_array)
        max_ineach = max(output)
        output = np.exp(output - max_ineach)
        output = output / np.sum(output)
        self.output[0][0] = output

    def backward(self, input_array, delta_array, learn_rate):
        self.delta_array = delta_array
        current_error_matrix = np.array(np.matrix(list(delta_array[0]) * self.weight.shape[1]).T)
        weight_update = (self.weight - np.array(list(input_array[0]) * self.weight.shape[0])) * current_error_matrix
        for i in range(10):
            weight_update[i] = (delta_array[0][0][i] - self.output[0][0][i]) * self.input_array
        self.weight -= learn_rate * weight_update
        _error = ((np.array(list(input_array[0]) * self.weight.shape[0]) - self.weight) * current_error_matrix).sum(axis=0)
        return _error.reshape(input_array.shape)

