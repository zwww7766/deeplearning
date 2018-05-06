#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from convlayer import *
from meanPooling import meanPoolingLayer
from FullConnectedLayer import FullConnectedLayer
from OutputLayer import OutputLayer
import numpy as np
import  json
from progressComponents import ShowProcess
import time
from PIL import Image

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
        self.outputlayer7 = OutputLayer(10, 84)
        self.layers = [self.convlayer1, self.poolinglayer2,\
                       self.convlayer3, self.poolinglayer4, self.convlayer5, self.fclayer6, self.outputlayer7]

    def showImg(self, array, p, cur):
        """卷积过程可视化，需要使用 unconv,unpooling 继续学习"""
        print  "layer: %i",cur
        array = np.array(array, dtype='Int32')
        pic = Image.fromarray(array)
        file_name = './deeplearning/traindata/pic/train/image%s.png' % (p+""+cur)
        # img.convert('L').save(file_name)
        # img.show()
        pic.convert('L').save(file_name)

    def forward(self, sample):
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output_array


        return output

    def backward(self, sample, label, learn_rate):
        #
        # label = np.array(label).reshape(len(label), 1)#10,1 与 偏差项统一
        # # np.where 三元操作符
        # # output_label[0][0][np.where(label>=np.max(label))[0][0]] = 1
        # # 取得是 输出的 最大值的index
        # output = sample
        # # 最外层的fc层的delta由label 和 output确定
        # output_label = self.layers[-1].activator.backward( self.layers[-1].output_array ) * (label - self.layers[-1].output_array)
        #
        # for layer in self.layers[::-1]:# output_label 为误差项 &^l
        #     layer.backward(output, output_label, learn_rate)
        #     output_label = layer.delta_array
        #     output = layer.output_array


        label = np.array(label).reshape(len(label), 1)#10,1 与 偏差项统一
        # np.where 三元操作符
        # output_label[0][0][np.where(label>=np.max(label))[0][0]] = 1
        # 取得是 输出的 最大值的index
        # output = sample
        # # 最外层的fc层的delta由label 和 output确定
        # output_label = self.layers[-1].activator.backward( self.layers[-1].output_array ) * (label - self.layers[-1].output_array)
        output_label = np.zeros([1,1,10])
        output_label[0] = label.T
        # print np.shape(output_label)
        self.outputlayer7.backward(self.fclayer6.output_array, output_label, learn_rate)
        self.fclayer6.backward(self.convlayer5.output_array, self.outputlayer7.delta_array, learn_rate)
        self.convlayer5.backward(self.poolinglayer4.output_array, self.fclayer6.delta_array, learn_rate)
        self.poolinglayer4.backward(self.convlayer3.output_array, self.convlayer5.delta_array, learn_rate)
        self.convlayer3.backward(self.poolinglayer2.output_array, self.poolinglayer4.delta_array, learn_rate)
        self.poolinglayer2.backward(self.convlayer1.output_array, self.convlayer3.delta_array, learn_rate)
        self.convlayer1.backward(sample, self.poolinglayer2.delta_array, learn_rate)

        # for layer in self.layers[::-1]:# output_label 为误差项 &^l
        #     layer.backward(output, output_label, learn_rate)
        #     output_label = layer.delta_array
        #     output = layer.output_array

    def train(self, labels, data_set, rate, epoch):
        """     训练函数
                labels: 样本标签
                data_set: 输入样本
                rate: 学习速率
                epoch: 训练轮数
        """
        for i in range(epoch):
            # process_bar = None
            # process_bar = ShowProcess(len(data_set))  # 1.在循环前定义类的实体， max_steps是总的步数
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],
                                      data_set[d], rate)
                print '%s/%s'%(d,len(data_set))
                # process_bar.show_process()  # 2.显示当前进度
                # time.sleep(0.05)
            # process_bar.close('done')
    def predict(self, data):
        return self.forward(data)

    def train_one_sample(self, label, sample, rate):
        res = self.forward(sample)
        # 32 x 32

        self.backward(sample, label, rate)
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
        """
        写入json
        :return:
        """
        fc7w = self.outputlayer7.weight
        fc7w = np.array(fc7w)
        self.npSave(fc7w,'fc7w')
        # fc7b = self.outputlayer7.b
        # fc7b = np.array(fc7b)
        # self.npSave(fc7b,'fc7b')
        #         --
        fc6w = self.fclayer6.W
        fc6w = np.array(fc6w)
        self.npSave(fc6w,'fc6w')
        fc6b = self.fclayer6.b
        fc6b = np.array(fc6b)
        self.npSave(fc6b,'fc6b')

        #         --
        c5w ,c5b = self.getFilter(self.convlayer5)
        p4w ,p4b = self.getFilter(self.poolinglayer4)
        c3w ,c3b = self.getFilter(self.convlayer3)
        p2w ,p2b = self.getFilter(self.poolinglayer2)
        c1w ,c1b = self.getFilter(self.convlayer1)
        c5w = np.array(c5w)
        c5b = np.array(c5b)
        self.npSave(c5w,'c5w')
        self.npSave(c5b,'c5b')
        p4w = np.array(p4w)
        p4b = np.array(p4b)
        self.npSave(p4w,'p4w')
        self.npSave(p4b,'p4b')
        c3w = np.array(c3w)
        c3b = np.array(c3b)
        self.npSave(c3w,'c3w')
        self.npSave(c3b,'c3b')
        p2w = np.array(p2w)
        p2b = np.array(p2b)
        self.npSave(p2w,'p2w')
        self.npSave(p2b,'p2b')
        c1w = np.array(c1w)
        c1b = np.array(c1b)
        self.npSave(c1w,'c1w')
        self.npSave(c1b,'c1b')
        file_name = './json_file.txt'
        # contain = [[fc7w, fc7b],[fc6w, fc6b],[c5w, c5b],[p4w, p4b],[c3w, c3b],[p2w, p2b],[c1w, c1b]]
        # contain = np.array(contain)

        # with open(file_name, 'w') as file_obj:
        #     json.dump(contain, file_obj)
    def npSave(self, v,n):
        np.save("./deeplearning/traindata/npdate/"+n+".npy", v)

    def npLoad(self, n):
        return np.load("./deeplearning/traindata/npdate/" + n + ".npy")
    def readJsonFile(self):
        """
        读取json文件
        """
        # self.outputlayer7.W = self.npLoad('fc7w')
        # self.outputlayer7.b = self.npLoad('fc7b')
        #         --
        self.fclayer6.W = self.npLoad('fc6w')
        self.fclayer6.b = self.npLoad('fc6b')
        #         --
        c5w = self.npLoad('c5w')
        c5b = self.npLoad('c5b')
        p4w = self.npLoad('p4w')
        p4b = self.npLoad('p4b')
        c3w = self.npLoad('c3w')
        c3b = self.npLoad('c3b')
        p2w = self.npLoad('p2w')
        p2b = self.npLoad('p2b')
        c1w = self.npLoad('c1w')
        c1b = self.npLoad('c1b')
        self.setFilter(c5w, c5b, self.convlayer5)
        self.setFilter(p4w, p4b, self.poolinglayer4)
        self.setFilter(c3w, c3b, self.convlayer3)
        self.setFilter(p2w, p2b, self.poolinglayer2)
        self.setFilter(c1w, c1b, self.convlayer1)
