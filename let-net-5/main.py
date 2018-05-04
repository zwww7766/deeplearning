# -*- coding: utf-8 -*-
import Loader
import network
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
    net = network.ConvNetwork()
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
            print 'convergence fail stop train: ',error
            net.save_filter_data()
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
        label = get_result(test_labels[i])
        # sample = np.array(test_data_set[i]).reshape(len(test_data_set[i]), 1)
        res = network.predict(test_data_set[i])
        print '%s,%s'%(label,np.argmax(res))
        if label != np.argmax(res):
            error += 1
    return float(error) / float(total)

def get_result(vec):
    max_value_index = float(0)
    max_value = float(0)
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def showImg(array):
    npa = np.array(array, dtype='Int32')
    pic = Image.fromarray(npa)
    pic.show()
    # 后续跟进存储操作