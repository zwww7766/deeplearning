# -*- coding:utf8 -*-
from ConvLayer import *
from PoolingLayer import *
from FullconnectedLayer import *
from Outputlayer import *

class ConvNet(object) :
    def __init__(self):
        convlayer3sizes = [[3, 5, 5]] * 6
        convlayer3sizes.extend([[4, 5, 5]] * 9)
        convlayer3sizes.extend([[6, 5, 5]])

        convlayer3choose_array = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 1], \
                                  [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 2],
                                  [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5], [0, 1, 2, 3, 4, 5]]

        self.convlayer1 = ConvLayer([[28, 28]] * 6, [[1, 5, 5]] * 9)
        self.poolinglayer2 = PoolingLayer([[14, 14]] * 6, [[2,2]] * 6)
        self.convlayer3 = ConvLayer([[10, 10]] * 16, convlayer3sizes, convlayer3choose_array)
        self.poolinglayer4 = PoolingLayer([[5, 5]] * 16, [[2, 2]] * 16)

        self.convlayer5 = ConvLayer([[1, 1]] * 120, [[16, 5, 5]] * 120)
        self.fclay6 = FullconnectedLayer(84, 120)

        self.outputlay7 = OutputLayer(10, 84)

    def forward(self, sample):
        self.convlayer1.forward(sample)
        self.poolinglayer2.forward(self.convlayer1.output)
        self.convlayer3.forward(self.poolinglayer2.output, True)
        self.poolinglayer4.forward(self.convlayer3.output)
        self.convlayer5.forward(self.poolinglayer4.output)
        self.fclay6.forward(self.convlayer5.output)
        self.outputlay7.forward(self.fclay6.output)
        return self.outputlay7.output[0][0]

    def backward(self, sample, label, learn_rate):
        output_error = np.zeros([1, 1, 10])
        output_error[0][0][label] = 1
        fclayer_error = self.outputlay7.backward(self.fclay6.output, output_error, learn_rate)
        cov5_error = self.fclay6.backward(self.convlayer5.output, fclayer_error, learn_rate)
        pool4_error = self.convlayer5.backward(self.poolinglayer4.output, cov5_error, learn_rate)
        cov3_error = self.poolinglayer4.backward(self.convlayer3.output, pool4_error, learn_rate)
        pool2_error = self.convlayer3.backward(self.poolinglayer2.output, cov3_error, learn_rate)
        cov1_error = self.poolinglayer2.backward(self.convlayer1.output, pool2_error, learn_rate)
        layer_error = self.convlayer1.backward(sample, cov1_error, learn_rate)

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
                print '%s/%s'%(d,len(data_set))

    def train_one_sample(self, label, sample, rate):
        self.forward(sample)
        # 32 x 32
        self.backward(sample, label, rate)

    def predict(self, data):
        return self.forward(data)