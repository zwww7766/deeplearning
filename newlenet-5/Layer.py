# -*- coding:utf8 -*-

import numpy as np

class Layer(object):
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.output = []
        for out_size in layer_size:
            self.output.append(np.zeros(out_size))
        self.output = np.array(self.output)
