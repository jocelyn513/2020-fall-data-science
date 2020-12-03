import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
def load_mnist(path="data/MNIST_data"):
    """
    导入mnist数据集
    :param path: 数据路径
    :return: 数据字典：{train_x,train_y,test_x,test_y}
    """

    mnist = input_data.read_data_sets(path, one_hot=True)

    data_dict = {
        "train_x": mnist.train.images,
        "train_y": mnist.train.labels,
        "test_x": mnist.test.images,
        "test_y": mnist.test.labels
    }
    return data_dict

def read_img2numpy(batch_size=64,img_h=64,img_w=64,path="data/faces"):
    """
    读取磁盘图像，并将图片重新调整大小
    :param batch_size: 每次读取图片数量
    :param img_h: 图片重新调整高度
    :param img_w: 图片重新调整宽度
    :param path: 数据存放路径
    :return: 图像numpy数组
    """
    file_list = os.listdir(path) # 图像名称列表
    data = np.zeros([batch_size,img_h,img_w,3],dtype=np.uint8) # 初始化numpy数组
    mask = np.random.choice(len(file_list),batch_size,replace=True)
    for i in range(batch_size):
        mm = Image.open(path+"/"+file_list[mask[i]])
        tem =mm.resize((img_w,img_h))  # 重新调整图片大小
        data[i,:,:,:] = np.array(tem)
    # 数据归一化 -1-1
    data = (data-127.5)/127.5 #-1-1
    return data

def img2gif(img_path="out/dcgan/",gif_path="out/dcgan/"):
    #获取图像文件列表
    file_list = os.listdir(img_path)
    imges = []
    for file in file_list:
        if file.endswith(".png"):
            img_name = img_path +file
            imges.append(imageio.imread(img_name))
    imageio.mimsave(gif_path+"result.gif",imges,fps=2)

if __name__ == '__main__':
    data = read_img2numpy()
    print(data.shape)
    data = ((data * 127.5) + 127.5).astype(np.uint8)
    plt.imshow(data[10])
    plt.show()
