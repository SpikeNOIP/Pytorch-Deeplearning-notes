"""
损失函数
    分类：
        1.二分类交叉熵损失函数
        2.多分类交叉熵损失函数
    回归：
        1.MSE：均方误差损失函数
        2.MAE：平均绝对误差损失函数
        3.Smooth L1: 结合上述两个的特点，升级，优化

多分类交叉熵损失：CrossEntropyLoss
公式：Loss = -ΣyLog(S(f(x)))
简单记忆：
        x : 特征
        f(x): 经过神经元计算后的结果
        S(f(x)): softmax映射成概率
        y: 样本x属于某一个类别的 真实概率

概率解释：
    损失函数结果 = 最小化正确类别所对应的预测概率的对数的负值 （非正确类别会被y=0给清洗掉 只留下正确的类别）
细节：
    CrossEntropyLoss = Softmax() + 损失计算,后续如果用nn.CrossEntropyLoss()则不需要再加Softmax()
"""

import torch
import torch.nn as nn

def dm01():
    #设置一个真实值
    #y_true = torch.tensor([[0,1,1],[1,0,0]])
    y_true = torch.tensor([1,0])

    #设置一个预测值
    y_pred = torch.tensor([[0.1,0.8,0.1],[0.7,0.2,0.1]],requires_grad=True)

    #创建多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    loss = criterion(y_pred, y_true)

    print(f"损失值：{loss}")

if __name__ == '__main__':
    dm01()