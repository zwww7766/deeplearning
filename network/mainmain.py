# -*- coding: UTF-8 -*-
import Network
import Loader
import time
import numpy as np
import matplotlib.pyplot as plt
import socket.config as config

global network

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


# 错误率


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


def predict_sample(sample):
    return network.predict(sample)


def train_and_evaluate(sample=[]):
    print '开始训练--->'
    config['status'] = '神经网络正在训练'
    epoch = 0
    x = []
    y = []
    train_data_set, train_labels = Loader.get_training_data_set()
    test_data_set, test_labels = Loader.get_test_data_set()
    network = Network.Network([784, 110, 10])
    # network = Network.Network([784, 300, 10])
    while True:
        epoch += 1
        # print 'train once------->train_data : %d 个，labels : %d 个' % (len(train_data_set), len(test_labels))

        network.train(train_labels, train_data_set, 0.1, 1)
        print '%s epoch %d finished' % (time.time(), epoch)
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print '%s after epoch %d, error ratio is %f' % (time.time(), epoch, error_ratio)
            x.append(error_ratio)
            y.append(epoch)
            if error_ratio > last_error_ratio:
                print '<---训练终止'
                config['status'] = '神经网络就绪'
                break
            else:
                last_error_ratio = error_ratio
            # print x
            # print y
            # plt.plot(y, x)
            # plt.xlabel('epoch / times')
            # plt.ylabel('error_rate / 1')
            # plt.title('train rate')
            # plt.savefig('./rate.png')
            # if error_ratio > last_error_ratio:

            # if error_ratio < 0.1:
            #     print '收敛成功，停止训练'
            #     last_error_ratio = error_ratio
            #     break
            # else:
            #     print '一次训练结束'



if __name__ == '__main__':
    train_and_evaluate()
