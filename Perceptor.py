# -*- coding:utf8 -*-

'''
神经网络感知器
'''
class Preceptron(object):
    def __init__(self,input_num,activator):

        '''

        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        '''
        self.activator = activator
        #权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        #构建一个长度为input_num的double类型list

        #偏置项初始化为0
        self.bias = 0.0
    def __str__(self):
        '''
        print时的附加输出项
        打印学习到的权重、偏置项
        '''
        return 'weigths\t:%s\nbias\t:%f\n' %(self.weights, self.bias)
        #预测
    def predict(self, input_vec):
        '''
        输入向量，输出感知器计算结果
        '''
        # input_vec[] 和 weight[] 打包在一起
        # 变成[(x1,w1),(x2,w2).....(xn,wn)]
        # 最后利用reduce求和
        # zip()多个数组打包方法
        # map(a,b)执行a函数b为参数
        return self.activator(
            reduce(lambda a, b: a+b,
                   map(lambda (x,w): x * w,
                       zip(input_vec, self.weights))
            , 0.0) + self.bias)
    def train(self , input_vecs, labels, iteration, rate):
        '''
        输入训练数据：一组向量、与每个向量对应的label 、训练轮数 、学习率
        '''
        # iteration:迭代
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        '''一次迭代。把所有训练数据过一遍'''
        # 把输入和输出打包在一起，成为样本的列表
        # 每个训练样本是（input_vex, label）
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        print'---------'
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            print '__________input%s',input_vec
            output = self.predict(input_vec)
            print 'predict out: %s:%s'%(output,label)
            # 更新权重
            self._update_weights(input_vec, output, label ,rate)

    def _update_weights(self, input_vec, output, label ,rate):
        '''按照感知器规则更新权重'''
        # 把input_vec 和 weigths 打包在一起
        delta = label - output
        self.weights = map(
            lambda (x, w): w + rate * delta * x,
            zip(input_vec, self.weights))

        self.bias +=rate * delta
