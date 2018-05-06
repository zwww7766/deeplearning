# -*- coding:utf8 -*-

from Layer import *

class FullconnectedLayer(Layer):
    def __init__(self, output_size, input_size):
        Layer.__init__(self, [[1, output_size]])
        Fi = input_size + 1
        self.weight = np.random.uniform(-2.4/Fi, 2.4/Fi, [output_size, input_size]) #84,120
        self.bias = np.random.uniform(-2.4/Fi, 2.4/Fi, [output_size]) #84

    def fc_op(self, pre_maps, node_index) :
            # pre_maps: 1x1x120
            pre_nodes = pre_maps.reshape([pre_maps.shape[0] * pre_maps.shape[1] * pre_maps.shape[2]]) #120
            val  = np.sum(self.weight[node_index] * pre_nodes) + self.bias[node_index] #
            val = np.exp((4.0/3)*val)
            self.output[0][0][node_index] = 1.7159 *  (val -1) / (val + 1)

    def forward(self, input) :
            for i in range(len(self.output[0][0])) : #84
                    self.fc_op(input, i)

    def backward(self, input_array, delta_array, learn_rate) :
        self.delta_array = delta_array
        # &^l * 本层输出 -->&
        pdelta_array = [((2.0/3)*(1.7159 - (1/1.7159) * self.output[0][0][i]**2))*delta_array[0][0][i]\
            for i in range(self.output.shape[-1])]
        # 1x84 > 84x1 dot 1x120 > 84x120
        weight_update = np.dot(np.matrix(pdelta_array).T, \
            np.matrix(input_array.reshape([1, input_array.shape[0] * input_array.shape[1] * input_array.shape[2]])))

        bias_update = np.array(pdelta_array)

        self.weight -= learn_rate * weight_update
        self.bias -= learn_rate * bias_update
        _error = np.array(np.dot(np.matrix(pdelta_array), np.matrix(self.weight))).reshape(input_array.shape)
        return _error



