import tensorflow as tf
import numpy as np
import DataHandler as dh


class DeepNet(object):

    def __init__(self, input_dim=784, hidden_layer=[100, 100], out_dim=10, lr=0.001, std=0.1):
        """
        初始化深度神经网络模型
        :param input_dim: 输入数据维度
        :param hiden_layer: 隐藏层数量
        :param out_dim: 输出维度
        :param lr: 学习率
        :param std: 标准差
        """
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.out_dim = out_dim
        self.lr = lr
        self.std = std
        self._init_net()  # 初始化网络结构
        self._init_train_method()  # 初始化训练方法

    def _init_variable(self, shape, name):
        """
        初始化变量
        :param shape:变量形状
        :param name: 变量名称
        :return: 返回初始化后变量
        """
        return tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=shape, stddev=self.std), name=name)

    def _init_net(self):
        """
        初始化网络结构
        :return:
        """
        # 构造数据特征值与目标值的占位符
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name="input_dim")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim], name="labels")
        # 构造网络结构
        # 构造输入层
        w = self._init_variable(shape=[self.input_dim, self.hidden_layer[0]], name="w0")
        b = self._init_variable(shape=[self.hidden_layer[0]], name='b0')
        affine = tf.matmul(self.x, w) + b  # x*w+b
        hidden = tf.nn.relu(affine)
        # 构造隐藏层
        for i in range(len(self.hidden_layer) - 1):
            with tf.name_scope("hidden_layer_" + str(i + 1)):
                w = self._init_variable(shape=[self.hidden_layer[i], self.hidden_layer[i + 1]],
                                        name="w_" + str(i + 1))
                b = self._init_variable(shape=[self.hidden_layer[i + 1]], name="b_" + str(i + 1))
                affine = tf.matmul(hidden, w) + b
                hidden = tf.nn.relu(affine)
        # 构造输出层
        w = self._init_variable(shape=[self.hidden_layer[-1], self.out_dim], name="w_out")
        b = self._init_variable(shape=[self.out_dim], name="b_out")
        self.out_scores = tf.matmul(hidden, w) + b

    def _init_train_method(self):
        """
        初始化训练方法，损失函数，梯度下降，Session
        :return:
        """
        # 构造损失函数
        with tf.name_scope("cross_entropy"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out_scores, labels=self.y))
        # 构造正确率计算
        with tf.name_scope("acc"):
            self._predict = tf.argmax(self.out_scores, 1)
            correct_predict = tf.equal(tf.argmax(self.out_scores, 1), tf.argmax(self.y, 1), name="correct_predict")
            self.acc = tf.reduce_mean(tf.cast(correct_predict, tf.float32)) \
 \
                # 构造优化方法
        with tf.name_scope("optimizer"):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # 初始化Session
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=1)

    def train(self, data_dict, itrs=10000, batch_size=100):
        """
        训练网络模型
        :param data_dict:数据字典
        :param itrs: 迭代次数
        :param batch_size: 批量大小，每次采样数据量
        :return:
        """
        for i in range(itrs):
            mask = np.random.choice(data_dict['train_x'].shape[0], batch_size, replace=True)
            batch_x = data_dict['train_x'][mask]
            bacth_y = data_dict['train_y'][mask]

            self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y: bacth_y})

            if i % 1000 == 0:
                temp_loss, temp_acc = self.test(data=data_dict['test_x'], labels=data_dict['test_y'])
                print("迭代", i, "次，当前损失", temp_loss, ",当前正确率", temp_acc)
                self.save()
        self.sess.close()

    def test(self, data, labels):
        """
        测试网络模型
        :param data:测试数据
        :param labels: 测试数据真实目标值
        :return:
        """
        temp_loss, temp_acc = self.sess.run([self.loss, self.acc], feed_dict={self.x: data, self.y: labels})
        return temp_loss, temp_acc

    def predict(self, data):
        """
        预测数据目标值
        :param data: 输入数据
        :return: 预测目标值
        """
        predict = self.sess.run(self._predict, feed_dict={self.x: data})
        return predict

    def save(self, path="model/dnn/"):
        """
        保存训练好的模型
        :param path:保存模型路径
        :return:
        """
        self.saver.save(sess=self.sess, save_path=path)

    def restore(self, path="model/dnn/"):
        """
        从磁盘恢复模型
        :param path: 恢复模型路径
        :return:
        """
        self.saver.restore(sess=self.sess, save_path=path)


if __name__ == '__main__':
    data_dict = dh.load_mnist()
    deepNet = DeepNet(input_dim=784, out_dim=10)
    deepNet.restore()
    deepNet.train(data_dict=data_dict, itrs=20000)
