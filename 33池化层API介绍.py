"""
池化层(Pooling Layer)：降维 降采样
池化层的API：nn.MaxPool2d() nn.AvgPool2d()

    nn.MaxPool2d()：最大池化
    取这个感受野中的最大值 作为输出

    nn.AvgPool2d()：平均池化
    取这个感受野中的平均值 作为输出

卷积层会改变输入图片的通道数，池化层不会改变输入图片的通道数
一般会选择最大池化，因为最大池化能保留突出特征
"""
import torch
import torch.nn as nn


#单通道的池化层
def dm01():
    #创建一个单通道的卷积层
    conv = torch.tensor(
        [
         [
             [0,1,2],
             [3,4,5],
             [6,7,8],
         ]
        ]
    )
    #创建最大池化层
    #参1：感受野的大小 参2：步长 参3：填充
    pool = nn.MaxPool2d(kernel_size=2,stride=1,padding=0)
    print(pool(conv))

    #创建平均池化层
    # 参1：感受野的大小 参2：步长 参3：填充
    pool = nn.AvgPool2d(kernel_size=2,stride=1,padding=0)
    print(pool(conv))

#多通道的池化层
def dm02():
    #创建一个多通道的卷积层
    conv = torch.tensor(
        [
         [
             [0,1,2],
             [3,4,5],
             [6,7,8],
         ],
         [
             [10,20,30],
             [40,50,60],
             [70,80,90],
         ],
         [
             [11,22,33],
             [44,55,66],
             [77,88,99],
         ]
        ]
    )
    #创建最大池化层
    #参1：感受野的大小 参2：步长 参3：填充
    pool = nn.MaxPool2d(kernel_size=2,stride=1,padding=0)
    print(pool(conv))

    #创建平均池化层
    # 参1：感受野的大小 参2：步长 参3：填充
    pool = nn.AvgPool2d(kernel_size=2,stride=1,padding=0)
    print(pool(conv))

if __name__ == '__main__':
    dm01()
    dm02()
