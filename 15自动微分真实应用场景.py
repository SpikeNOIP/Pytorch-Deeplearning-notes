"""
演示自动微分的场景

1.先做前向传播 (正向传播)，计算出预测值 (z)
2.基于损失函数，结合 预测值 和 真实值 来计算梯度
3.结合权重更新公式 w新 = w旧 - 学习率 * 梯度 更新权重

"""
import torch

#1 定义 y 表示特征
x = torch.ones(2,5)
print(f"x:{x},shape:{x.shape}")

#2 定义 y 表示：标签（真实值），假设：2行3列 全0矩阵
y = torch.zeros(2,3)
print(f"y:{y},shape:{y.shape}")

#3 初始化（可自动微分）模板 权重和偏置
w = torch.randn(5,3,requires_grad=True)
print(f"w:{w},shape:{w.shape}")

b = torch.randn(3,requires_grad=True)
print(f"b:{b},shape:{b.shape}")

#4 前向传播（正向传播） 计算出 预测值z
z = torch.matmul(x,w) + b
print(f"z:{z},shape:{z.shape}")

#5 定义损失函数
criterion = torch.nn.MSELoss()
loss = criterion(z,y)

#6 反向传播（计算梯度）更新参数
loss.sum().backward()
print(f"w的梯度：{w.grad}")
print(f"b的梯度：{b.grad}")

w.data = w.data - 0.01 * w.grad
b.data = b.data - 0.01 * b.grad
print(f"w更新后的值：{w}")
print(f"b更新后的值：{b}")
