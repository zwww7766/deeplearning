# -*- coding: UTF-8 -*-
# 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算。
import math
import numpy as np

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

class Node(object):
    def __init__(self, layer_index, node_index):
        """
        构造节点对象。
        layer_index: 节点所属的层的编号
        node_index: 节点的编号
        """
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        """
        设置节点的输出值。如果节点属于输入层会用到这个函数。
        """
        self.output = output

    def append_downstream_connection(self, conn):
        """
        添加一个到下游节点的连接
        """
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        """
        添加一个到上游节点的连接
        """
        self.upstream.append(conn)

    def calc_output(self):
        """
        根据式1计算节点的输出
        ret为 每次运算结果的赋值对象
        每次从self.upstream中获取connection对象的output
        由于是向量的集合，下面的reduce函数，做了一次向量内所有output乘以weight 的和，作为一个对下层节点的输出
        """

        try:
            output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
            self.output = sigmoid(output)
            # self.output =  1.0 / (1 + math.exp(-output))
        except :
            print '错误节点的详细信息--->:',output
            #上层是785 因为784 加一个偏置项

            # for a in self.upstream:
                # print a.weight
                # print a.upstream_node.output
                # print a.weight * a.upstream_node.output
            # print '错误信息over---->'

    def calc_hidden_layer_delta(self):
        """
        节点属于隐藏层时，根据式4计算delta
        对所有下游节点的 weight 乘以 delta(误差项) 求和
        """
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        """
        节点属于输出层时，根据式3计算delta
        """
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        """
        打印节点的信息
        """
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str
