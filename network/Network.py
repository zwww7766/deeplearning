# -*- coding: UTF-8 -*-
import Connections
import Layer
import Connection
import FullConnectedLayer
import numpy as np
# Sigmoid激活函数类


class SigmoidActivator(object):

    def forward(self, weighted_input):

        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        # print np.shape(output)
        # print np.shape(1 - output)
        return output * (1 - output)


class Network(object):
    def __init__(self, layers):

        """
            构造函数
                """
        self.count = 0
        self.layers = []
        for i in range(len(layers) - 1):
            print '-------------add fc layer'
            self.layers.append(
                FullConnectedLayer.FullConnectedLayer(layers[i], layers[i + 1],
                    SigmoidActivator()
                )
            )
        # """
        # 初始化一个全连接神经网络
        # layers: 二维数组，描述神经网络每层节点数
        # """
        #
        # self.connections = Connections.Connections()
        # self.layers = []
        # layer_count = len(layers)
        # node_count = 0
        # #构造层
        # for i in range(layer_count):
        #     self.layers.append(Layer.Layer(i, layers[i]))
        #
        #     """
        #     第一层connection 在 0层和一层之间 取 layer = 0
        #     第二层connection 在 1层和2层之间 取 layer = 1
        #     connection 层数 是 layer的数量减1
        #     """
        # #构造链接层---为con绑定与之相关的上层节点和下层节点
        # for layer in range(layer_count - 1):
        #     #使用两个for in 取出 上下游泳节点， 交给 connection 初始化，并存储对象 作为list的一个元素
        #     connections = [Connection.Connection(upstream_node, downstream_node)
        #                    for upstream_node in self.layers[layer].nodes
        #                    for downstream_node in self.layers[layer + 1].nodes[:-1]]
        #     #给链接的上游节点绑定下游节点，链接的下游节点绑定上游节点
        #     for conn in connections:
        #         self.connections.add_connection(conn)
        #         conn.downstream_node.append_upstream_connection(conn)
        #         conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, data_set, rate, epoch):
        # """
        # 训练神经网络
        # labels: 数组，训练样本标签。每个元素是一个样本的标签。
        # data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
        # """
        # k = 0
        # for i in range(iteration):
        #     for d in range(len(data_set)):
        #         k =k+ 1
        #         print k/60000
        #         self.train_one_sample(labels[d], data_set[d], rate)
        """     训练函数
                labels: 样本标签
                data_set: 输入样本
                rate: 学习速率
                epoch: 训练轮数
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.count += 1
                # print self.count
                self.train_one_sample(labels[d],
                                      data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        label = np.array(label).reshape(len(label), 1)
        sample = np.array(sample).reshape(len(sample), 1)
        # """
        # 内部函数，用一个样本训练网络
        # """
        # self.predict(sample)
        # self.calc_delta(label)
        # self.update_weight(rate)
        # print '------------>1'
        self.predict(sample)
        # print '------------>2'
        self.calc_gradient(label)
        # print '------------>3'
        self.update_weight(rate)

    def calc_delta(self, label):
        """
        内部函数，计算每个节点的delta
        """
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        # """
        # 内部函数，更新每个连接权重
        # """
        # for layer in self.layers[:-1]:
        #     for node in layer.nodes:
        #         for conn in node.downstream:
        #             conn.update_weight(rate)
        for layer in self.layers:
            layer.update(rate)

    def calc_gradient(self, label):

        # print np.shape(self.layers[-1].output)
        # print np.shape(self.layers[-1].activator.backward(
        #     self.layers[-1].output
        # ))
        # print np.shape(label - self.layers[-1].output)
        # print np.shape(label)
        # print np.shape(self.layers[-1].output)
        # print np.shape(1-self.layers[-1].output)
        # print 'out put layer clac'
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        # print np.shape(delta)
        # print '----->start hidden layer'
        for layer in self.layers[::-1]:
            # print 'hidden layer clac'
            layer.backward(delta)
            delta = layer.delta
        return delta
        # """
        # 内部函数，计算每个连接的梯度
        # """
        # for layer in self.layers[:-1]:
        #     for node in layer.nodes:
        #         for conn in node.downstream:
        #             conn.calc_gradient()

    def get_gradient(self, label, sample):
        """
        获得网络在一个样本下，每个连接上的梯度
        label: 样本标签
        sample: 样本输入
        """

        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        # """
        # 根据输入的样本预测输出值
        # sample: 数组，样本的特征，也就是网络的输入向量
        # """
        # self.layers[0].set_output(sample)
        # for i in range(1, len(self.layers)):
        #     self.layers[i].calc_output()
        # return map(lambda node: node.output, self.layers[-1].nodes[:-1])

        output = sample
        # print np.shape(output)
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def dump(self):
        """
        打印网络信息
        """
        for layer in self.layers:
            layer.dump()

    def gradient_check(network, sample_feature, sample_label):
        """
            梯度检查
            network: 神经网络对象
            sample_feature: 样本的特征
            sample_label: 样本的标签
            """
        # 计算网络误差
        network_error = lambda vec1, vec2: \
            0.5 * reduce(lambda a, b: a + b,
                         map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                             zip(vec1, vec2)))
        # 获取当前样本下每个链接的梯度
        network.get_gradient(sample_feature, sample_label)

        # 对每个权重做梯度检查
        for conn in network.connections.connections:
            # 获取指定连接的梯度
            actual_gradient = conn.get_gradient()
            # 增加一个很小的值，计算网络的误差
            epsilon = 0.0001
            conn.weight += epsilon
            error1 = network_error(network.predict(sample_feature), sample_label)
            # 减去一个很小的值，计算网络的误差
            conn.weight -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
            error2 = network_error(network.predict(sample_feature), sample_label)
            # 根据式6计算期望的梯度值
            expected_gradient = (error2 - error1) / (2 * epsilon)
            # 打印
            print 'expected gradient: \t%f\nactual gradient: \t%f' % (
                expected_gradient, actual_gradient)
