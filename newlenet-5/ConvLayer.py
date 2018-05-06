# -*- coding:utf8 -*-

from Layer import *

class ConvLayer(Layer):
    def __init__(self, layer_size = [], cov_core_sizes = [], combine = []):
        Layer.__init__(self, layer_size)
        self.covcores = []
        self.covbias = []
        self.combine = combine

        #初始化卷积核
        for covcore in cov_core_sizes:
            Fi = covcore[0] * covcore[1] + 1
            # 随机卷积核
            self.covcores.append(np.random.uniform(-2.4/Fi, 2.4/Fi, covcore))
            # 随机卷积bias
            self.covbias.append(np.random.uniform(-2.4/Fi, 2.4/Fi))

        self.covcores = np.array(self.covcores)

    def cov_op(self, input, index):
        """卷积操作"""
        input_shape = input.shape
        covcore_shape = self.covcores[index].shape
        output_shape = self.output[index].shape

        #检查是否匹配
        if not(output_shape[-2] == input_shape[-2] - covcore_shape[-2] + 1
           and output_shape[-1] == input_shape[-1] - covcore_shape[-1] + 1):
            return None

        for i in range(output_shape[-2]):
            for j in range(output_shape[-1]):

                recp = input[: ,i:i + covcore_shape[-2], j:j + covcore_shape[-1] ]
                val = np.sum(recp * self.covcores[index]) + self.covbias[index]

                val = np.exp((4.0/3) * val)
                self.output[index][i][j] = 1.7159 * (val-1) /(val+1)

    def forward(self, input_array, c3 = []):
        """forward运算"""
        for i in range(len(self.output)):
            if not  c3:
                self.cov_op(input_array, i)
            else:
                self.cov_op(input_array[self.combine[i]], i)

    def backward(self, input_array, delta_array, learn_rate):
        """反向计算"""
        self.delta_array = delta_array

        # 本层的 输出尺寸 = 本层的误差项
        output_line = self.output.reshape([self.output.shape[0] * self.output.shape[1] * self.output.shape[2]])
        delta_array_line = delta_array.reshape([delta_array.shape[0] * delta_array.shape[1] * delta_array.shape[2]])

        # pdelta_array 为 &^l * activator.forward( net^l-1 )
        pdelta_array = np.array([((2.0/3)*(1.7159 - (1/1.7159) * output_line[i]**2)) * delta_array_line[i]\
                                 for i in range(len(output_line))]).reshape(self.output.shape)

        # 结果重置
        weight_update = self.covcores * 0
        bias_update = np.zeros([len(self.covbias)])
        _error = np.zeros(input_array.shape)
        for i in range(self.output.shape[0]):

            if self.combine != []:
                select_input_array = input_array[self.combine[i]]
                select_error = _error[self.combine[i]]

            else:
                select_input_array = input_array
                select_error = _error
            # list[:, 1:4, 2:5] 按照深度和每层宽度距离拷贝数组,单独 : 为遍历这一层
            #
            # c5 : output 120,1,1   covcores 120,16,5,5   input 16,5,5   & 16,5,5
            # c3          16,10,10           6,3,5,5            6,14,14    6,14,14
            # c1          6,28,28            6,1,5,5            1,32,32    1,32,32

            # 本层 w 的期望 = & * a^l-1  :也就是用本层 & 作为卷积核，在input上进行cross-correlation
            # 本层 bias 的期望 的和
            for mi in range(self.output.shape[1]):
                for mj in range(self.output.shape[2]):
                    cov_maps = select_input_array[:, mi:mi+self.covcores[i].shape[1], mj:mj+self.covcores[i].shape[2]]
                    weight_update[i] += cov_maps * pdelta_array[i][mi][mj]
                    bias_update[i] += pdelta_array[i][mi][mj]
                    # 使用第d个filter对第l层相应的第d个sensitivity map进行卷积，
                    # 得到一组N个l-1层的偏sensitivity map。依次用每个filter做这种卷积，
                    # 就得到D组偏sensitivity map。最后在各组之间将N个偏sensitivity map 按元素相加，得到最终的N个l-1层的sensitivity map：
                    select_error[:, mi:mi+self.covcores[i].shape[1], mj:mj+self.covcores[i].shape[2]]\
                        += self.covcores[i] * pdelta_array[i][mi][mj]

        self.covcores -= learn_rate * weight_update
        self.covbias -= learn_rate * bias_update
        return _error








