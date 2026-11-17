"""
实现自动微分 循环计算梯度 更新参数
    loss = x**2 + 20
"""
import torch

w = torch.tensor(10,requires_grad=True,dtype=torch.float)

loss = w**2 + 20       #求导： loss' = 2w

print(f"开始 权重初始值：{w},(0.01*w.grad):无，loss:{loss}")

#利用梯度下降算法 循环迭代100次求最优解
for i in range(1,101):
    loss = w ** 2 + 20

    if w.grad is not None:
        w.grad.zero_()
    #如果不清零 这里的梯度会叠加 梯度默认是会叠加的

    loss.sum().backward()

    w.data = w.data - 0.01 * w.grad

    print(f"第{i}次梯度更新结果,梯度值为：{w.grad:.5f}，loss:{loss:.5f}")

print(f"权重：{w},梯度：{w.grad:.5f},loss:{loss:.5f}")