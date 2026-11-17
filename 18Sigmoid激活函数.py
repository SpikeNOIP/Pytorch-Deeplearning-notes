"""
绘制激活函数Sigmoid的 函数图像 和 导数图像
激活函数目的：
    给模型增加非线性功能，让模型（神经元）既可以分类 也可以做回归
激活函数的分类：
    1.线性激活函数：线性回归
    2.非线性激活函数：Sigmoid、Tanh、ReLU、LeakyReLU、PReLU、ELU、SELU、GELU

    Sigmoid激活函数：
        主要应用于二分类的输出层，且适用于 浅层神经网络（不超过5层）
        数据主要在 [-6,6]之间有效果， 在 [-3,3]之间效果明显
        求导后范围在 [0, 0.25] 之间 在5层 之后 更新效率降低 

"""
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

fig,axes = plt.subplots(1,2)

#生成 -20 - 20 之间的随机1000个数据点
x = torch.linspace(-20,20,1000) #这里的steps参数默不是步长，而是参数个数

#计算上述1000个点，Sigmoid激活函数处理后的值
y=torch.sigmoid(x)
print(f"y:{y}")

axes[0].plot(x,y)
axes[0].set_title("Sigmoid激活函数图像")
axes[0].grid()

#在第二个图上 绘制Sigmoid激活函数的倒数图像
x = torch.linspace(-20,20,1000,requires_grad=True)

torch.sigmoid(x).sum().backward()

axes[1].plot(x.detach(),x.grad) #返回一个新张量，数据和原来的 x 共享 但把 requires_grad 关掉（变成 False），并切断和计算图的联系
axes[1].set_title("Sigmoid激活函数导数图像")
axes[1].grid()
plt.show()