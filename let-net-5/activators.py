# -*- coding: utf-8 -*-
import numpy as np


class ReluActivator(object):
    def forward(self, weighted_input):
        #return weighted_input
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0


class IdentityActivator(object):
    def forward(self, weighted_input):
        # return weighted_input
        val = np.exp((4.0/3) * weighted_input)
        return 1.7159 * (val - 1) / (val + 1)

    def backward(self, output):
        return 1
        # return (2.0/3) * (1.7159 - (1/1.7159) * output**2)

class SigmoidActivator(object):
    def forward(self, weighted_input):
        res = 1.0 / (1.0 + np.exp(-weighted_input))
        # val = np.exp((4.0 / 3) * weighted_input)
        # return np.longfloat(1.7159 * (val - 1) / (val + 1))
        return res

    def backward(self, output):
        # return np.longfloat(2.0/3 * (1.7159 - (1/1.7159) * output**2))
        return output * (1 - output)

class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output