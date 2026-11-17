"""
批量归一化，Batch Normalization(BN),用于环节模型过拟合的问题
思路：
    现堆数据做标准化（会丢失一些信息），然后再对数据做缩放(λ-> 权重) 和 平移(β，->偏置)，来补回这些确实的信息
    使用在计算机视觉领域较多 对图像数据有很好的处理效果

    1. 在每一层的输入之前添加一层批量归一化层
    2. 在批量归一化层之后添加一个线性层
    3. 在线性层之后添加一个激活函数层

    BatchNorma1d: 主要应用在全连接层之后或处理一维数据的网络中，接受（N，num_features）例如文本处理
    BatchNorm2d: 主要应用在卷积层之后或处理二维数据的网络中，(N,C,H,W)例如图像处理
    BatchNorm3d: 主要应用在三维数据的神经网络(3D CNN)中，例如处理视频或医学图像，接受形状为（N,C,D,H,W）的张量输入

"""
import torch
import torch.nn as nn

def dm_01():
    #准备图像数据
    #size=(N,C,H,W) N:样本数量(图片数量) C:通道数量(RGB就是三通道 只用两个颜色就是双通道) H:高度像素 W:宽度像素
    pic_data = torch.randn(size=(1,2,3,4))
    print(f"pic_data:{pic_data}")

    """
    BatchNorm2d 里的 momentum 是干什么的
    它控制的是滑动平均(exponential moving average)的更新系数,用来维护两个"移动统计量":running_mean(移动均值)和 running_var(移动方差)。
    
    # ... existing code ...
    bn2d = nn.BatchNorm2d(num_features=2,eps=1e-05, momentum=0.1, affine=True)

    # 训练模式下，第一次前向
    out1 = bn2d(pic_data)
    print("第一次 running_mean:", bn2d.running_mean)
    # 第二次前向（momentum=0.1 → 旧的占0.9，新batch占0.1）
    pic_data2 = torch.randn(size=(1,2,3,4))
    out2 = bn2d(pic_data2)
    print("第二次 running_mean:", bn2d.running_mean)
    print("可学习参数 gamma(weight):", bn2d.weight, "\n可学习参数 beta(bias):", bn2d.bias)
    # ... existing code ...
    """
    #批量归一化层
    #参数1: num_features: 就是图片的通道数
    #参数2: eps: 小常数 防止除0错误
    #参数3: momentum: 动量参数 默认0.1 这里的动量和SGD优化梯度下降不是一个东西
    #参数4: affine: 是否使用可学习的参数λ和β 默认True
    bn2d = nn.BatchNorm2d(num_features=2,eps=1e-05, momentum=0.1, affine=True)

    output_bn2d = bn2d(pic_data)
    print(f"output_bn2d:{output_bn2d}")

def dm_02():
    pic_data = torch.randn(size=(3, 4))
    print(f"pic_data:{pic_data}")

    linear = nn.Linear(4,3)
    output_linear = linear(pic_data)
    print(f"output_linear:{output_linear}")

    #创建一维批量归一化层只用传入特征数量作为参数就好
    bn1d = nn.BatchNorm1d(num_features=3)
    bn1d_output = bn1d(output_linear)
    print(f"bn1d_output:{bn1d_output}")

if __name__ == '__main__':
    #dm_01()
    dm_02()