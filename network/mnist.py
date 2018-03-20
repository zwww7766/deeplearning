#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import struct


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
        f = open(self.path, 'rb')#二进制读
        content = f.read()
        f.close()
        return content

    def to_int(self, byte):
        """
        将unsigned byte字符转换为整数
        """
        return struct.unpack('B', byte)[0]


class ImageLoader(Loader):

    def get_picture(self, content, index):

        '''
        内部函数，从文件中获取图像
        '''

        start = index * 28 * 28 + 16
        picture = []
        # pic_b = b''
        for i in range(28):
            picture.append([])
            for j in range(28):
                # pic_b += content[start + i * 28 + j]
                picture[i].append(
                    # content[start + i * 28 + j])
                    self.to_int(content[start + i * 28 + j]))
        # file_name   = '../traindata/pic/%s.jpg' % index
        # file_object = open(file_name, 'w')
        # file_object.write(pic_b)
        # file_object.close()
        # print len(pic_b)
        return picture

    def get_one_sample(self, picture, index):
        """
        内部函数，将图像转化为样本的输入向量
        """
        c = ''
        for i in picture:
            for j in i:
                c +=str(j)
        # n = np.array(picture)
        # print  n
        img = Image.fromarray(n)
        file_name = '../traindata/pic/'+index+'.png'
        img.save(file_name)
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
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
                    self.get_picture(content, index), index))
        return data_set
# 标签数据加载器
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print '魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols)

    image_size = num_rows * num_cols
    # +16
    # print '--------header_length-->:',struct.calcsize(fmt_header)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print '已解析 %d' % (i + 1) + '张'
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
        # +784
        # print 'images_size---->',struct.calcsize(fmt_image)
        img = Image.fromarray(images[i])
        file_name = '../traindata/pic/%i.png' % i
        # Image.open(file_name).convert('RGB').save(file_name)
        # if i <=100:
        img.convert('L').save(file_name)
        # else :
    return images



if  __name__ == '__main__':
    # image_loader = ImageLoader('../traindata/train-images-idx3-ubyte', 60000)
    # v = open('../traindata/pic/tt.png', 'rb').read()
    # for i in v:
    #     print struct.unpack('B', i)[0]
    # image_loader.load()
    decode_idx3_ubyte('../traindata/train-images-idx3-ubyte')