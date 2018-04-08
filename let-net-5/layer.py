# -*- coding: UTF-8 -*-
import numpy as np


class Layer(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.output_array = []
        for mapsize in sizes:
            self.output_array.append(np.zeros(mapsize))
        self.output_array = np.array(self.output_array)
