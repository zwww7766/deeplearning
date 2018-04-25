# -*- coding: UTF-8 -*-
from convlayer import *
from meanPooling import meanPoolingLayer
from FullConnectedLayer import FullConnectedLayer
import numpy as np
import  json
from progressComponents import ShowProcess
import time

class ConvNetwork(object):
    def __init__(self):
        convlayer3sizes = [[3, 5, 5]] * 6
        convlayer3sizes.extend([[4, 5, 5]] * 9)
        convlayer3sizes.extend([[6, 5, 5]])

        convlayer3choose_array = [[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,0],[5,0,1],\
                [0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,0],[4,5,0,1],[5,0,1,2],[0,1,3,4],[1,2,4,5],[0,2,3,5],[0,1,2,3,4,5]]
        self.convlayer1 = ConvLayer([[28, 28]] * 6, [[1, 5, 5]] * 6)
        self.poolinglayer2 = meanPoolingLayer([[14, 14]] * 6, [[2, 2]] * 6)

        self.convlayer3 = ConvLayer([[10, 10]] * 16, convlayer3sizes, convlayer3choose_array)
        self.poolinglayer4 = meanPoolingLayer([[5, 5]] * 16, [[2, 2]] * 16)

        self.convlayer5 = ConvLayer([[1, 1]] * 120, [[16, 5, 5]] * 120)
        self.fclayer6 = FullConnectedLayer(84, 120)
        # self.convlayer6 = ConvLayer([[1, 1]] * 84, [[1, 1]] * 84)
        self.outputlayer7 = FullConnectedLayer(10, 84)
        # self.convlayer7 = ConvLayer([[1, 1]] * 10, [[1, 1]] * 10)
        self.layers = [self.convlayer1, self.poolinglayer2,\
                       self.convlayer3, self.poolinglayer4, self.convlayer5, self.fclayer6, self.outputlayer7]
    def forward(self, sample):
        output = sample

        for layer in self.layers:
            layer.forward(output)
            output = layer.output_array
        return output

    def backword(self, sample, label, learn_rate):

        # output_label = np.zeros([1, 1, 10])
        # print output_label[0][0]
        label = np.array(label).reshape(len(label), 1)#10,1 与 偏差项统一
        # np.where 三元操作符
        # output_label[0][0][np.where(label>=np.max(label))[0][0]] = 1
        # 取得是 输出的 最大值的index
        output = sample
        # 最外层的fc层的delta由label 和 output确定
        output_label = self.layers[-1].activator.backward(
            self.layers[-1].output_array
        ) * (label - self.layers[-1].output_array)
        for layer in self.layers[::-1]:
            # output_label 为误差项 &^l
            layer.backward(output, output_label, learn_rate)
            output_label = layer.delta_array
            output = layer.output_array

    def train(self, labels, data_set, rate, epoch):
        """     训练函数
                labels: 样本标签
                data_set: 输入样本
                rate: 学习速率
                epoch: 训练轮数
        """
        for i in range(epoch):
            process_bar = ShowProcess(100)  # 1.在循环前定义类的实体， max_steps是总的步数
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],
                                      data_set[d], rate)
                process_bar.show_process(d)  # 2.显示当前进度
                time.sleep(0.05)
            process_bar.close('done')
    def predict(self, data):
        return self.forward(data)
    def train_one_sample(self, label, sample, rate):
        self.forward(sample)
        # 32 x 32
        self.backword(sample, label, rate)
    def getFilter(self,layer):
        a = []
        b = []
        for filter in layer.filters:
            a.append(filter.weights)
            b.append(filter.bias)
        return a,b

    def setFilter(self, a, b, layer):
        for i in range(len(layer.filters)):
            layer.filters[i] = a[i]
            layer.bias[i] = b[i]

    def save_filter_data(self):
        fc7w = self.outputlayer7.W
        fc7b = self.outputlayer7.b
#         --
        fc6w = self.fclayer6.W
        fc6b = self.fclayer6.b
#         --
        c5w ,c5b = self.getFilter(self.convlayer5)
        p4w ,p4b = self.getFilter(self.poolinglayer4)
        c3w ,c3b = self.getFilter(self.convlayer3)
        p2w ,p2b = self.getFilter(self.poolinglayer2)
        c1w ,c1b = self.getFilter(self.convlayer1)

        file_name = './json_file.txt'
        contain = [[fc7w, fc7b],[fc6w, fc6b],[c5w, c5b],[p4w, p4b],[c3w, c3b],[p2w, p2b],[c1w, c1b]]
        # nums = {"name": "Mike", "age": 12}
        with open(file_name, 'w') as file_obj:
            '''写入json文件'''
            json.dump(contain, file_obj)

    def readJsonFile(self):
        file_name = './json_file.txt'
        with open(file_name) as file_obj:
            '''读取json文件'''
            contain = json.load(file_obj)  # 返回列表数据，也支持字典

            self.outputlayer7.W = contain[0][0]
            self.outputlayer7.b = contain[0][1]
            #         --
            self.fclayer6.W = contain[1][0]
            self.fclayer6.b = contain[1][1]
            #         --
            self.setFilter(contain[2][0], contain[2][1], self.convlayer5)
            self.setFilter(contain[3][0], contain[3][1], self.poolinglayer4)
            self.setFilter(contain[4][0], contain[4][1], self.convlayer3)
            self.setFilter(contain[5][0], contain[5][1], self.poolinglayer2)
            self.setFilter(contain[6][0], contain[6][1], self.convlayer1)
