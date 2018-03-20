# -*- coding: UTF-8 -*-
from convlayer import *
from meanPooling import meanPoolingLayer
from FullConnectedLayer import FullConnectedLayer
import numpy as np

class ConvNetwork(object):
    def __init__(self):
        convlayer3sizes = [[3, 5, 5]] * 6
        convlayer3sizes.extend([[4, 5, 5]] * 9)
        convlayer3sizes.extend([[6, 5, 5]])

        convlayer3choose_array = [[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,0],[5,0,1],\
                [0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,0],[4,5,0,1],[5,0,1,2],[0,1,3,4],[1,2,4,5],[0,2,3,5],[0,1,2,3,4,5]]
        print("-------step1")
        self.convlayer1 = ConvLayer([[28, 28]] * 6, [[1, 5, 5]] * 6)
        self.poolinglayer2 = meanPoolingLayer([[14, 14]] * 6, [[2, 2]] * 6)

        self.convlayer3 = ConvLayer([[10, 10]] * 16, convlayer3sizes, convlayer3choose_array)
        self.poolinglayer4 = meanPoolingLayer([[5, 5]] * 16, [[2, 2]] * 16)

        self.convlayer5 = ConvLayer([[1, 1]] * 120, [[16, 5, 5]] * 120)
        self.fclayer6 = FullConnectedLayer(84, 120)

        self.outputlayer7 = FullConnectedLayer(10, 84)

        self.layers = [self.convlayer1, self.poolinglayer2,\
                       self.convlayer3, self.poolinglayer4, self.convlayer5, self.fclayer6, self.outputlayer7]

    def forward(self, sample):
        output = sample

        for layer in self.layers:
            layer.forward(output)
            output = layer.output_array
        return output

    def backword(self, sample, label, learn_rate):
        output_label = np.zeros([1, 1, 10])
        output_label[0][0][label] = 1
        output = sample
        for layer in self.layers:
            layer.backward(output, output_label, learn_rate)
            output_label = layer.delta
            output = layer.output_array

    def train(self, labels, data_set, rate, epoch):
        """     训练函数
                labels: 样本标签
                data_set: 输入样本
                rate: 学习速率
                epoch: 训练轮数
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],
                                      data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        print '----- foward labels -----'
        self.forward(sample)
        print '----- backward labels -----'
        self.backword(sample, label, 0.1)
