"""
图像的分类：
    二值图：    单通道图像，黑白图像，0和1表示颜色
    灰度图：    单通道图像，灰度图像，0-255表示颜色
    彩色图RGB： 三通道图像，RGB图像，0-255表示颜色
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

def dm01():
    #定义一个全黑白的图片
    #H:高度HEIGHT W:宽度WEIGHT C:通道数CONNECTIONS
    img = np.zeros((640,640,3))
    print(f"shapes:{img.shape}")
    #绘制图片
    plt.imshow(img)
    plt.show()

    #定义一个全白色的图片
    img = np.ones((640,640,3))
    print(f"shapes:{img.shape}")
    #绘制图片
    plt.imshow(img)
    plt.show()

    #绘制一个随机颜色的图片
    img = np.random.random((640,640,3))
    print(f"shapes:{img.shape}")
    #绘制图片
    plt.imshow(img)
    plt.show()

    #也可以用torch绘图
    img = torch.full((640,640,3),188)
    print(f"shapes:{img.shape}")
    #绘制图片
    plt.imshow(img)
    plt.show()


def dm02():
    img1 = plt.imread("D:\\DeepLearning\\imageData\\TOKO.png")
    print(f"shapes:{img1.shape}")
    plt.imshow(img1)

    plt.imsave("D:\\DeepLearning\\imageData\\TOKO_save.png",img1)
    plt.show()


if __name__ == '__main__':
    dm02()
