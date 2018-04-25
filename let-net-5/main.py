# -*- coding: UTF-8 -*-
import Loader
import network
import numpy as np
from PIL import Image
global net
def train_and_evaluate():
    print '-------step 0-------'
    train_data_set, train_labels = Loader.get_training_data_set()
    test_data_set, test_labels = Loader.get_test_data_set()
    print '-------step 1-------'
    net = network.ConvNetwork()
    print '-------step 2-------'
    error = 1
    while(True):
        print 'train num: %s labels: %s'%(len(train_data_set),len(train_labels))
        net.train(train_labels, train_data_set, 0.1, 1)
        newerror = evaluate(net,train_data_set,test_labels)
        if newerror < error:
            error = newerror
            print 'once useful train'
            print error
        else:
            print 'convergence fail stop train '
            break

def tran_once(data):
     val =  net.predict(data)
     print 'predict result：',val
     return val


if __name__ == '__main__':
    print 'network ready start'
    train_and_evaluate()


def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        sample = np.array(test_data_set[i]).reshape(len(test_data_set[i]), 1)
        predict = get_result(network.predict(sample))
        if label != predict:
            error += 1
    return float(error) / float(total)

def get_result(vec):
    max_value_index = float(0)
    max_value = float(0)
    for i in range(len(vec)):
        # print '-------%d-------' % i
        # print vec[i]
        # print max_value
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def showImg(array):
    npa = np.array(array, dtype='Int32')
    pic = Image.fromarray(npa)
    pic.show()
    # 后续跟进存储操作