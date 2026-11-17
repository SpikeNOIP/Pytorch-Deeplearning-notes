"""
案例：
    演示自动微分模块，具体如何求导

回顾：
    W新 = W旧 - 学习率 * 梯度
    梯度 = 损失函数的导数

    Pytorch内置自动微分模块，专门实现针对于损失函数的求导
    结合 反向传播 更新 权重参数W 和 偏置参数b

细节：
    只有表量张量才能求导。
    在自动微分的时候推荐使用浮点数据
"""
import torch

#参1：初始值，参2：是否自动求导(微分) 如果不标记这个 反向传播会报错，参3：数据类型
w = torch.tensor(10,requires_grad=True,dtype=torch.float)

#定义 损失函数 loss
loss = 2 * w ** 2       #求导 loss = 4w

#loss.backward()
loss.sum().backward()
#这里如果loss是个矩阵/向量，得先用.sum() 或 .mean() 压成一个数 只能对一个表量张量求微分
#沿着计算图反向走一遍，算出"loss 对每个 requires_grad=True 的变量的导数"，并把结果存进变量的 .grad 里。
#并不会主动更新w 只会更新 .grad()

w.data = w.data - 0.01 * w.grad
#为什么不自动做？ 因为有很多更新方法 用户可以自己选择

print(f"更新后的权重：{w}")    #更新后的权重：9.600000381469727

