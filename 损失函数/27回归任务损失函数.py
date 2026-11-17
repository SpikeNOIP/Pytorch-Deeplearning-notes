"""
回归任务常用损失函数如下：
    1.MSE：均方误差损失函数
    2.MAE：平均绝对误差损失函数
    3.Smooth L1: 结合上述两个的特点，升级，优化

MAE:    平均绝对误差
    公式：
        Loss = Σ|y - f(x)|
        类似与L1正则化，权重可以降为0 数据会变稀疏
        在0点不平滑，可能错过最小值

MSE:    均方误差
    公式：
        Loss = Σ(y - f(x))^2
        类似与L2正则化，权重不会降为0 数据不会变稀疏
        在0点平滑，没有可能错过最小值

        在误差特别大的情况下 会出现 梯度更新爆炸的情况，导致模型无法训练

Smooth L1: 结合上述两个的特点，升级，优化
    基于 MAE MSE的综合 平滑了0点 同时避免更新爆炸
    采用分段函数：[-1,1]采用L2 [-1,1]之外采用L1
"""
import torch
import torch.nn as nn

def dm01():
    #准备数据
    y_true = torch.tensor([1.0,1.0,2.0],dtype=torch.float32)
    y_pred = torch.tensor([0.8,0.9,0.1],requires_grad=True)

    criterion = nn.L1Loss()

    loss = criterion(y_pred, y_true)

    print(loss.detach().numpy())

def dm02():
    #准备数据
    y_true = torch.tensor([1.0,1.0,2.0],dtype=torch.float32)
    y_pred = torch.tensor([0.8,0.9,0.1],requires_grad=True)

    criterion = nn.MSELoss()

    loss = criterion(y_pred, y_true)

    print(loss.detach().numpy())

def dm03():
    #准备数据
    y_true = torch.tensor([1.0,1.0,2.0],dtype=torch.float32)
    y_pred = torch.tensor([0.8,0.9,0.1],requires_grad=True)

    criterion = nn.SmoothL1Loss()

    loss = criterion(y_pred, y_true)

    print(loss.detach().numpy())

if __name__ == '__main__':
    #dm01()
    #dm02()
    dm03()