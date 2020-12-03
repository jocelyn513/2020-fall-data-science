import tensorflow as tf
import numpy as np
import DataHandler as dh
from tensorflow.contrib.layers import batch_norm


class CNN(object):
    def __init__(self, input_dim=784, out_dim=10, lr=0.01, std=0.1):
        """
        初始化卷积神经网络
        :param input_dim: 输入数据维度
        :param out_dim: 输出维度
        :param lr: 学习率
        :param std: 标准差
        """
        self.x_dim = input_dim
        self.out_dim = out_dim
        self.lr = lr
        self.std = std
        self._init_net()  # 初始化卷积神经网络结构
        self._init_train_methods()  # 初始化网络训练方法

    def _init_net(self):
        """
        初始化卷积神经网络结构
        :return:
        """
        # 构造占位符
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.x_dim], name="input_data")  # [n,m]
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim], name="labels")
        # 重塑输入tensor
        self.input_img = tf.reshape(self.x, shape=[-1, 28, 28, 1])
        # 构造第一层卷积
        with tf.name_scope("layer_1"):
            layer_dim_1 = 50
            # w=[3,3,1,layer_dim_1]对应着 layer_dim_1个3*3卷积核
            w = self._init_varible(shape=[3, 3, 1, layer_dim_1], name='w')
            b = self._init_varible(shape=[layer_dim_1], name="b")
            conv = tf.nn.conv2d(input=self.input_img, filter=w, strides=[1, 1, 1, 1], padding="SAME") + b
            # 输出形状：[28,28,layer_dim1]
            conv = batch_norm(conv)  # 输出形状：[28,28,layer_dim1]
            active = tf.nn.relu(conv)  # 输出形状：[28,28,layer_dim1]
            # ksize=[1,2,2,1]对应着2*2池化
            active = tf.nn.max_pool(value=active, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # 输出形状：[14,14,layer_dim1]

        with tf.name_scope("layer_2"):
            layer_dim_2 = 50
            w = self._init_varible(shape=[3, 3, layer_dim_1, layer_dim_2], name='w')
            b = self._init_varible(shape=[layer_dim_2], name='b')
            conv = tf.nn.conv2d(input=active, filter=w, strides=[1, 1, 1, 1], padding="SAME") + b
            # 输出形状：[14,14,layer_dim2]
            conv = batch_norm(conv)
            active = tf.nn.relu(conv)
            active = tf.nn.max_pool(value=active, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # 输出形状：[7,7,layer_dim2]
        with tf.name_scope("layer_3"):
            layer_dim_3 = 25
            w = self._init_varible(shape=[3, 3, layer_dim_2, layer_dim_3], name='w')
            b = self._init_varible(shape=[layer_dim_3], name='b')
            conv = tf.nn.conv2d(input=active, filter=w, strides=[1, 1, 1, 1], padding="SAME") + b
            # 输出形状：[7,7,layer_dim_3]
            conv = batch_norm(conv)
            active = tf.nn.relu(conv)
        # 构造输出层
        with tf.name_scope("out_layer"):
            hidden_out = tf.reshape(active, shape=[-1, 7 * 7 * layer_dim_3])  # 将tensor重塑为[m行，n列]
            w = self._init_varible(shape=[7 * 7 * layer_dim_3, self.out_dim], name="w_out")
            b = self._init_varible(shape=[self.out_dim], name="b_out")
            self.out_scores = tf.matmul(hidden_out, w) + b

    def _init_train_methods(self):
        """
        初始化网络训练方法：损失函数，梯度下降，Session
        :return:
        """
        # 初始化损失函数
        with tf.name_scope("cross_entropy"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out_scores, labels=self.y))
        # 构造正确率计算
        with tf.name_scope("acc"):
            self._predict = tf.argmax(self.out_scores, 1)
            correct_predict = tf.equal(tf.argmax(self.out_scores, 1), tf.argmax(self.y, 1), name="correct_predict")
            self.acc = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        # 构造梯度训练方法
        with tf.name_scope("Adam"):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # 初始化Session
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=1)

    def train(self, data_dict, itrs=10000, batch_size=50):
        """
        训练网络
        :param data_dict: 训练数据字典
        :param itrs: 训练迭代次数
        :param batch_size: 采样数据大小
        :return:
        """
        for i in range(itrs):
            mask = np.random.choice(data_dict['train_x'].shape[0], batch_size, replace=True)
            batch_x = data_dict['train_x'][mask]
            batch_y = data_dict['train_y'][mask]
            # 训练模型
            self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y})
            # 验证模型
            if i % 1000 == 0:
                tem_loss, tem_acc = self.test(data=data_dict['test_x'], labels=data_dict['test_y'])
                print("迭代：", i, "次,当前损失值：", tem_loss, " 当前正确率：", tem_acc)
                self.save()
        self.sess.close()

    def test(self, data, labels):
        """
        测试模型性能
        :param data: 数据特征值
        :param labels: 数据的目标值
        :return: 返回损失值与正确率
        """
        tem_loss,tem_acc = self.sess.run([self.loss,self.acc],feed_dict={self.x:data,self.y:labels})
        return tem_loss,tem_acc

    def save(self, path="model/cnn/"):
        """
        保存模型
        :param path:模型保存路径
        :return:
        """
        self.saver.save(self.sess,save_path=path)

    def restore(self, path="model/cnn/"):
        """
        恢复模型
        :param path: 模型保存路径
        :return:
        """
        self.saver.save(sess=self.sess,save_path=path)

    def predict(self, data):
        """
        预测输入数据目标值
        :param data: 输入数据特征值
        :return: 预测目标值
        """
        pre = self.sess.run(self._predict,feed_dict={self.x:data})
        return pre

    def _init_varible(self, shape, name):
        """
        初始化tensorflow变量
        :param shape: 变量形状
        :param name: 变量名称
        :return: tensorflow变量
        """
        return tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=shape, stddev=self.std), name=name)


if __name__ == '__main__':
    data_dict = dh.load_mnist()
    cnn = CNN(input_dim=784,out_dim=10)
    cnn.train(data_dict=data_dict,itrs=5000)