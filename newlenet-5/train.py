from numpy import *
from ConvNet import *
import time
import struct
import copy


def train_net(train_covnet, cycle, learn_rate, case_num=-1):
    # Read data
    # Change it to your own dataset path
    trainim_filepath = './deeplearning/traindata/train-images-idx3-ubyte'
    trainlabel_filepath = './deeplearning/traindata/train-labels-idx1-ubyte'
    trainimfile = open(trainim_filepath, 'rb')
    trainlabelfile = open(trainlabel_filepath, 'rb')
    train_im = trainimfile.read()
    train_label = trainlabelfile.read()
    im_index = 0
    label_index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', train_im, im_index)
    magic, numLabels = struct.unpack_from('>II', train_label, label_index)
    print ('train_set:', numImages)

    train_btime = time.time()
    # Begin to train
    for c in range(cycle):
        im_index = struct.calcsize('>IIII')
        label_index = struct.calcsize('>II')
        train_case_num = numImages
        if case_num != -1 and case_num < numImages:
            train_case_num = case_num
        for case in range(train_case_num):
            im = struct.unpack_from('>784B', train_im, im_index)
            label = struct.unpack_from('>1B', train_label, label_index)
            im_index += struct.calcsize('>784B')
            label_index += struct.calcsize('>1B')
            im = array(im)
            im = im.reshape(28, 28)
            bigim = list(ones((32, 32)) * -0.1)
            for i in range(28):
                for j in range(28):
                    if im[i][j] > 0:
                        bigim[i + 2][j + 2] = 1.175
            im = array([bigim])
            label = label[0]
            print (case, label)
            train_covnet.forward(im)

            train_covnet.backward(im, label, learn_rate[c])

    print ('train_time:', time.time() - train_btime)


def test_net(train_covnet, case_num=-1):
    # Read data
    # Change it to your own dataset path
    testim_filepath = './deeplearning/traindata/t10k-images-idx3-ubyte'
    testlabel_filepath = './deeplearning/traindata/t10k-labels-idx1-ubyte'
    testimfile = open(testim_filepath, 'rb')
    testlabelfile = open(testlabel_filepath, 'rb')
    test_im = testimfile.read()
    test_label = testlabelfile.read()

    im_index = 0
    label_index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', test_im, im_index)
    magic, numLabels = struct.unpack_from('>II', test_label, label_index)
    print('test_set:', numImages)
    im_index += struct.calcsize('>IIII')
    label_index += struct.calcsize('>II')

    correct_num = 0
    testcase_num = numImages
    if case_num != -1 and case_num < numImages:
        testcase_num = case_num

    # To test
    for case in range(testcase_num):
        im = struct.unpack_from('>784B', test_im, im_index)
        label = struct.unpack_from('>1B', test_label, label_index)
        im_index += struct.calcsize('>784B')
        label_index += struct.calcsize('>1B')
        im = array(im)
        im = im.reshape(28, 28)
        bigim = list(ones((32, 32)) * -0.1)
        for i in range(28):
            for j in range(28):
                if im[i][j] > 0:
                    bigim[i + 2][j + 2] = 1.175
        im = array([bigim])
        label = label[0]
        train_covnet.forward(im)
        print '%s,%s' % (argmax(train_covnet.outputlay7.output[0][0]), label)
        if argmax(train_covnet.outputlay7.output[0][0]) == label:
            correct_num += 1
    correct_rate = correct_num / float(testcase_num)
    print('test_correct_rate:', correct_rate)


def predict(train_covnet, p):

    testim_filepath = './deeplearning/traindata/t10k-images-idx3-ubyte'
    testlabel_filepath = './deeplearning/traindata/t10k-labels-idx1-ubyte'
    testimfile = open(testim_filepath, 'rb')
    testlabelfile = open(testlabel_filepath, 'rb')
    test_im = testimfile.read()
    test_label = testlabelfile.read()

    im_index = 0
    label_index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', test_im, im_index)
    magic, numLabels = struct.unpack_from('>II', test_label, label_index)
    print('train_set:', numImages)
    im_index += struct.calcsize('>IIII')
    label_index += struct.calcsize('>II')

    for i in range(p):
        im_index += struct.calcsize('>784B')
        label_index += struct.calcsize('>1B')


    im = struct.unpack_from('>784B', test_im, im_index)
    label = struct.unpack_from('>1B', test_label, label_index)
    im_index += struct.calcsize('>784B')
    label_index += struct.calcsize('>1B')
    im = array(im)
    im = im.reshape(28, 28)
    bigim = list(ones((32, 32)) * -0.1)
    for i in range(28):
        for j in range(28):
            if im[i][j] > 0:
                bigim[i + 2][j + 2] = 1.175
    im = array([bigim])
    label = label[0]
    train_covnet.forward(im)
    return argmax(train_covnet.outputlay7.output[0][0]),label
