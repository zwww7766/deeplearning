#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import struct
# from bp import *
from datetime import datetime
# 数据加载器基类
import  numpy as np

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
        self.load_progress()
        '''
        内部函数，从文件中获取图像
        '''
        #从0开始
        start = index * 28 * 28 + 16
        picture = []
        # pic_b = ''
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    # 按行i  j值逐步取单个字节
                    self.to_int(content[start + i * 28 + j]))
        return picture

    def get_one_sample(self, picture):
        """
        内部函数，将图像转化为样本的输入向量
        """
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        # 单个样本节点数784 已计算
        print sample
        return sample

    def load(self):
        """
        加载数据文件，获得全部样本的输入向量
        """
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
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
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        """
        内部函数，将一个值转换为10维标签向量
        """
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        #转为(10,)
        return label_vec


def get_training_data_set():
    """
    获得训练数据集
    """
    # image_loader = ImageLoader('../traindata/train-images-idx3-ubyte', 60000)
    # label_loader = LabelLoader('../traindata/train-labels-idx1-ubyte', 60000)
    image_loader = ImageLoader('./MyMachineLearning/traindata/train-images-idx3-ubyte', 60000)
    label_loader = LabelLoader('./MyMachineLearning/traindata/train-labels-idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()



def get_test_data_set():
    """
    获得测试数据集
    """
    # image_loader = ImageLoader('../traindata/t10k-images-idx3-ubyte', 10000)
    # label_loader = LabelLoader('../traindata/t10k-labels-idx1-ubyte', 10000)
    image_loader = ImageLoader('./MyMachineLearning/traindata/t10k-images-idx3-ubyte', 10000)
    label_loader = LabelLoader('./MyMachineLearning/traindata/t10k-labels-idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()

# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 37, 122, 174, 236, 189, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 60, 119, 186, 225, 254, 254, 254, 254, 254, 176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 120, 202, 254, 254, 254, 254, 254, 253, 218, 234, 254, 224, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 143, 237, 254, 254, 254, 254, 229, 133, 101, 43, 11, 185, 254, 200, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 184, 251, 254, 254, 251, 125, 32, 16, 0, 0, 2, 179, 254, 254, 82, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 237, 254, 187, 76, 19, 0, 0, 0, 0, 23, 182, 254, 245, 102, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 75, 70, 1, 0, 0, 0, 0, 0, 48, 162, 254, 254, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 198, 254, 251, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 148, 224, 254, 254, 250, 78, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 152, 241, 254, 254, 254, 254, 254, 254, 159, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 237, 254, 254, 254, 254, 207, 202, 254, 254, 254, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 132, 254, 254, 227, 113, 28, 3, 0, 56, 213, 254, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 204, 141, 9, 0, 0, 0, 0, 0, 156, 254, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 179, 254, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 234, 254, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 235, 254, 131, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 66, 206, 254, 216, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 96, 162, 254, 254, 188, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 126, 213, 254, 254, 254, 209, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 199, 243, 243, 243, 254, 254, 254, 255, 238, 128, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]