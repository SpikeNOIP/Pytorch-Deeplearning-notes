"""
卷积层的API, 用于 图像的局部特征 获取：特征图（Feature Map）

卷积神经网络(Convolutional Neural Network)
    组成：
        卷积层(Convolutional Layer)：用来提取图像的局部特征 每一个卷积核就是一个神经元，得到特征图
            卷积核(Convolutional Kernel)：
                1.卷积核的个数：feature_map个数
                2.卷积核的大小：卷积核的尺寸
                3.卷积核的参数：卷积核的权重参数
                4.卷积核的偏置：卷积核的偏置参数

        池化层(Pooling Layer)：降维 降采样

        全连接层(Fully Connected Layer)：用于预测结果并输出

    特征图的计算方式：
    N = (W - F + 2P) / S + 1
    N: 输出特征图的尺寸
    W: 输入特征图的尺寸
    F: 卷积核的尺寸
    P: 填充的尺寸
    S: 步长
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def dm01():
    img = plt.imread("D:\\DeepLearning\\imageData\\TOKO.png")
    # 转换图像为张量
    img_tensor = torch.tensor(img, dtype=torch.float32)
    #现在的图像HWC格式->转换成CWH格式
    #把通道数放到最前面
    img_tensor_CHW = torch.permute(img_tensor, (2, 0, 1))
    #现在的图像只有一张 我需要给图像增加一个维度 在第0维度 用来表示有几张图片
    img_tensor_CHW = torch.unsqueeze(img_tensor_CHW, 0)
    print(f"img_tensor_CHW.shape:{img_tensor_CHW.shape}")
    #[1, 3, 1536, 1536] 一张图片 3个通道 1536*1536

    #参1：输入的通道数 参2：输出通道数（卷积核的个数） 参3：卷积核的尺寸 参4：填充的尺寸
    conv = nn.Conv2d(3,3,5,1,padding=0)
    img_conv = conv(img_tensor_CHW)

    #print(f"img_conv.shape:{img_conv.shape}")
    #(W - F + 2P) / S + 1 = (1536 - 5 + 2*0) / 1 + 1 = 1532

    #获取四个特征图
    img2 = img_conv[0]

    #把CHW转换成WHC
    img2 = torch.permute(img2, (1, 2, 0))

    #显示第一维度的图像
    img3 = img2[:, :, 0].detach().numpy()
    print(f"img3.shape:{img3.shape}")
    plt.imshow(img3,cmap='gray')
    plt.show()

    #显示第二维度的图像
    img4 = img2[:,:,1].detach().numpy()
    print(f"img4.shape:{img4.shape}")
    plt.imshow(img4,cmap='gray')
    plt.show()

    #显示第三维度的图像
    img5 = img2[:,:,2].detach().numpy()
    print(f"img5.shape:{img5.shape}")
    plt.imshow(img5,cmap='gray')
    plt.show()

if __name__ == '__main__':
    dm01()


