# -*- coding: UTF-8 -*-
import layer
import numpy as np
class ConvLayer(layer):
    def __init__(self, sizes, filters, c3=[]):
        layer.__init__(self, sizes)
        self.filters = []
        self.bias = []
        for filter in filters:
            offset = filter[0] + filter[1] + 1
            self.filters.append(np.random.uniform(-2.4/offset, 2.4/offset, filter))
            self.bias.append(np.random.uniform(-2.4/offset, 2.4/offset))
        self.filters = np.array(self.filters)

    def conv(self):
        pass

    def calc_maps(self):
        pass
