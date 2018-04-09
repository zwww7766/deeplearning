# -*- coding: UTF-8 -*-
import Loader
import network
import numpy as np
from PIL import Image

def train_and_evaluate():
    print '-------step 0-------'
    epoch = 0
    train_data_set, train_labels = Loader.get_training_data_set()
    test_data_set, test_labels = Loader.get_test_data_set()
    print '-------step 1-------'
    net = network.ConvNetwork()
    print '-------step 2-------'

    epoch +=1
    print '训练数据量: %s labels: %s'%(len(train_data_set),len(train_labels))
    net.train(train_labels, train_data_set, 0.1, 1)

if __name__ == '__main__':
    print 'network ready start'
    train_and_evaluate()


def showImg(array):
    npa = np.array(array, dtype='Int32')
    pic = Image.fromarray(npa)
    pic.show()
    # 后续跟进存储操作