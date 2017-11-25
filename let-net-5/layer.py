# -*- coding: UTF-8 -*-
import numpy as np


class Layer(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.maps = []
        for mapsize in sizes:
            self.maps.append(np.zeros(mapsize))
        self.maps = np.array(self.maps)
