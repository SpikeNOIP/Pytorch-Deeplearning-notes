"""
正则化的作用：
    缓解模型过拟合的问题
    提高模型的泛化能力
    避免模型在训练集上过拟合，提高模型在测试集上的表现

正则化的方法：
    1. L1正则化 权重可以降为0
    2. L2正则化 权重可以无限接近于0
    3. Dropout随机失活
    4. BN批量归一化

Dropout:
    通常每一层隐藏层后面（激活函数之后）都可以加一层 Dropout，用于防止过拟合
     - 每个神经元以概率 p 被丢弃（置 0），比如 p=0.5 表示每个神经元有50% 概率被丢。未被丢弃的神经元会乘上一个缩放因子，缩放因子的计算公式为 1/(1-p)
     - 每次训练都是随机选一批神经元失活，不是固定的、也不是全部。
     - 因为每次随机丢弃不同神经元，相当于训练了很多个不同的"子网络"，最后效果等于它们的集成，从而缓解过拟合。
"""

import torch
import torch.nn as nn

#1.定义函数，随机失活
def dm01_dropout():
    #创建隐藏层输出结果 1行4列 0-10
    t1 = torch.randint(0,10,size=(1,4)).float()
    print(f"隐藏层：{t1}")

    #创建线性层接受 隐藏层出来的数据 4行1列
    linear = nn.Linear(4,4)
    l1 = linear(t1)
    print(f"加权求和结果：{l1}")

    #创建激活函数
    activation = nn.ReLU()
    l1out = activation(l1)
    print(f"激活函数结果：{l1out}")

    #创建Dropout层 p=0.5 每个神经元有百分之50的概率被kill掉
    dropout = nn.Dropout(p=0.5)
    l1out = dropout(l1out)
    print(f"Dropout结果：{l1out}")

if __name__ == '__main__':
    dm01_dropout()