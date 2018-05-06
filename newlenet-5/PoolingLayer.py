# -*- coding:utf8 -*-

from Layer import *

class PoolingLayer(Layer):
    def __init__(self, layer_size = [], pool_core_size = []):
        Layer.__init__(self, layer_size)
        Fi = pool_core_size[0][0] * pool_core_size[0][1] + 1
        self.poolparas = np.random.uniform(-2.4/Fi, 2.4/Fi, [len(layer_size), 2])
        self.poolcore_sizes = np.array(pool_core_size)

    def pool_op(self, pre_map, pool_index):
        """池运算操作"""
        pre_map_shape = pre_map.shape
        poolcore_size = self.poolcore_sizes[pool_index]
        for i in range(int(pre_map_shape[0] / poolcore_size[0])):
            for j in range(int(pre_map_shape[1] / poolcore_size[1])):
                val = self.poolparas[pool_index][0] * np.sum(pre_map[i * poolcore_size[0]:(i + 1) * poolcore_size[0], \
                                                          j * poolcore_size[1]:(j + 1) * poolcore_size[1]]) + \
                      self.poolparas[pool_index][1]
                val = np.exp((4.0 / 3) * val)
                self.output[pool_index][i][j] = 1.7159 * (val - 1) / (val + 1)

    def forward(self, input):
        """向前运算"""
        for i in range(len(self.output)):
            self.pool_op(input[i], i)

    def backward(self, input_array, delta_array, learn_rate,):
        """向后运算"""
        self.delta_array = delta_array
        output_line = self.output.reshape([self.output.shape[0] * self.output.shape[1] * self.output.shape[2]])
        delta_array_line = delta_array.reshape(
            [delta_array.shape[0] * delta_array.shape[1] * delta_array.shape[2]])

        pcurrent_error = np.array([((2.0 / 3) * (1.7159 - (1 / 1.7159) * output_line[i] ** 2)) * delta_array_line[i] \
                                for i in range(len(output_line))]).reshape(self.output.shape)

        weight_update = np.zeros([len(self.poolparas)])
        bias_update = np.zeros([len(self.poolparas)])
        _error = np.zeros(input_array.shape)

        for i in range(self.output.shape[0]):
            for mi in range(self.output.shape[1]):
                for mj in range(self.output.shape[2]):
                    weight_update[i] += pcurrent_error[i][mi][mj] * np.sum(
                        input_array[i][mi * self.poolcore_sizes[i][0]:(mi + 1) * self.poolcore_sizes[i][0], \
                        mj * self.poolcore_sizes[i][1]:(mj + 1) * self.poolcore_sizes[i][1]])
                    _error[i][mi * self.poolcore_sizes[i][0]:(mi + 1) * self.poolcore_sizes[i][0], \
                        mj * self.poolcore_sizes[i][1]:(mj + 1) * self.poolcore_sizes[i][1]] = \
                        pcurrent_error[i][mi][mj] * self.poolparas[i][0]
                bias_update[i] += np.sum(pcurrent_error[i])

            self.poolparas[:, 0:1] -= learn_rate * np.matrix(weight_update).T
            self.poolparas[:, 1:2] -= learn_rate * np.matrix(bias_update).T
        return _error