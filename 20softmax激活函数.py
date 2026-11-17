"""
Softmax激活函数：
    将多分类 的结果以概率形式展示，且概率和相加为1，最终选取概率值最大的分类作为结果

    记忆：如何选择激活函数：
    隐藏层：
        ReLU > LeakyReLU > PReLU > Tanh > Sigmoid
    输出层：
        二分类：Sigmoid
        多分类：softmax
        回归问题：identity
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

# scores = torch.tensor([0.2,0.3,0.4,0.5,0.6,0.7,0.8])
scores = torch.tensor([[0.1,0.2,0.3],[0.4,0.5,0.6]])

#参1：分数分布 参2：按照维度0 就是横着来 当多维时 就是竖着来 一个特征的来
probabilities = torch.softmax(scores,dim=0)
print(probabilities)

