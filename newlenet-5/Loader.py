#!/usr/bin/env python
# -*- coding: utf-8 -*-
import struct
from PIL import Image
# from bp import *
from datetime import datetime
# 数据加载器基类
import  numpy as np
from func import *
import os

class Loader(object):
    def __init__(self, path, count):
        """
        初始化加载器
        path: 数据文件路径
        count
        """
        self.path = path
        self.count = count
        self.progress = 0

    def get_file_content(self):
        """
        读取文件内容
        """
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

    def load_progress(self):
        self.progress += 1
        # print '%s' % self.progress

    def to_int(self, byte):
        """
        将unsigned byte字符转换为整数--一次一个
        """

        return struct.unpack('B', byte)[0]
# 图像数据加载器


class ImageLoader(Loader):
    def get_picture(self, content, index):
        # self.load_progress()
        '''
        内部函数，从文件中获取图像
        '''
        #从0开始
        start = index * 28 * 28 + 16
        picture = []

        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    # 按行i  j值逐步取单个字节
                    self.to_int(content[start + i * 28 + j]))
        return self.mix_pic(picture)

    def mix_pic(self, pic):
        mixp = list(np.ones((32, 32)) * -0.1)
        for i in range(28):
            for j in range(28):
                if pic[i][j] > 0:
                    mixp[i+2][j+2] = 1.175
        return np.array(mixp)

    def get_one_sample(self, picture):
        """
        内部函数，将图像转化为样本的输入向量
        """
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        # 单个样本节点数784 已计算
        return sample

    def load(self):
        """
        加载数据文件，获得全部样本的输入向量
        """
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                # self.get_one_sample(
                # 外padding两层，用来帮助收集图像边缘特征
                    np.array([np.array(self.get_picture(content, index))]))

            # )
        return data_set
# 标签数据加载器


class LabelLoader(Loader):
    def load(self):
        """
        加载数据文件，获得全部样本的标签向量
        """
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.to_int(content[index+8]))
        return labels

    def norm(self, label):
        """
        内部函数，将一个值转换为10维标签向量
        """
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(1)
            else:
                label_vec.append(0)
        #转为(10,)
        return label_vec


def get_training_data_set():
    """
    获得训练数据集
    """
    print 'get data'
    image_loader = ImageLoader('./deeplearning/traindata/train-images-idx3-ubyte', 100)
    label_loader = LabelLoader('./deeplearning/traindata/train-labels-idx1-ubyte', 100)
    # image_loader = ImageLoader('./MyMachineLearning/traindata/train-images-idx3-ubyte', 60000)
    # label_loader = LabelLoader('./MyMachineLearning/traindata/train-labels-idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()



def get_test_data_set():
    """
    获得测试数据集
    """
    image_loader = ImageLoader('./deeplearning/traindata/t10k-images-idx3-ubyte', 100)
    label_loader = LabelLoader('./deeplearning/traindata/t10k-labels-idx1-ubyte', 100)
    # image_loader = ImageLoader('./MyMachineLearning/traindata/t10k-images-idx3-ubyte', 10000)
    # label_loader = LabelLoader('./MyMachineLearning/traindata/t10k-labels-idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()

if __name__ == '__main__':
    testImgs , testLabel = get_test_data_set()
    print  len(testImgs)
    for i in range(len(testImgs)):
        img = Image.fromarray(np.array(testImgs[i], dtype='Int32'))
        val =testLabel[i].index(max(testLabel[i]))
        file_name = './deeplearning/traindata/pics/image%s.png' % i
        # img.show()
        img.convert('L').save(file_name)