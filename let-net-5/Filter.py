# -*- coding: utf-8 -*-
import numpy as np
import copy

class Filter(object):
    def __init__(self, type):
        Fi = type[0] * type[1] + 1
        self.weights = np.random.uniform(-2.4/Fi, 2.4/Fi, type)
        self.bias = np.random.uniform(-2.4/Fi, 2.4/Fi)
        self.weights_grad = np.zeros(
            self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        # print '-------+++--------learnrate:%s weightgrad:%s'%(learning_rate, self.weights_grad[0][0])
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad
        # print self.weights[0][0]

