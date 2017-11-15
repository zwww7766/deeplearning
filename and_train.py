# -*- coding:utf8 -*-

import  Perceptor
def f(x):
    '''
    定义一个激活函数
    '''
    return 1 if x > 0 else 0

def get_training_dataset():
    #基于真值表构建训练数据
    input_vecs= [[1,1],[0,0],[1,0],[0,1]]
    labels = [1,0,0,0]
    return input_vecs,labels

def train_percetron():
    p = Perceptor.Preceptron(2,f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return  p
if __name__ == '__main__':
    and_perception = train_percetron()

    print and_perception
    print '1 and 1 = %d' % and_perception.predict([1, 1])
    print '0 and 0 = %d' % and_perception.predict([0, 0])
    print '1 and 0 = %d' % and_perception.predict([1, 0])
    print '0 and 1 = %d' % and_perception.predict([0, 1])