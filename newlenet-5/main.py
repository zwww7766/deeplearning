# -*- coding: utf-8 -*-
import Loader
from ConvNet import *
import numpy as np
from PIL import Image
net = None
test_data_set = None
test_labels = None
def train_and_evaluate():
    print '-------step 0-------'
    global test_data_set
    global test_labels
    global net
    train_data_set, train_labels = Loader.get_training_data_set()
    test_data_set, test_labels = Loader.get_test_data_set()
    print '-------step 1-------'
    net = ConvNet()
    print '-------step 2-------'
    error = 1
    while(True):
        print 'train num: %s labels: %s'%(len(train_data_set),len(train_labels))
        net.train(train_labels, train_data_set, 0.0001, 1)
        newerror = evaluate(net,test_data_set,test_labels)
        if newerror < error:
            error = newerror
            print 'once useful train'
            print error
        else:
            print 'convergence fail stop train: '
            print error
            break

def tran_once(p):
     global test_data_set
     global test_labels
     global net
     data = test_data_set[p]
     val =  net.predict(data)
     print 'predict result：',val
     # print 'label:',test_labels[p]
     return val


if __name__ == '__main__':
    print 'network ready start'
    train_and_evaluate()


def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = test_labels[i]
        res = network.predict(test_data_set[i])
        print '%s,%s'%(label,np.argmax(res))
        if label != np.argmax(res):
            error += 1
    return float(error) / float(total)

def showImg(array):
    npa = np.array(array, dtype='Int32')
    pic = Image.fromarray(npa)
    pic.show()
    # 后续跟进存储操作