"""
7种参数初始化方法

参数初始化的目的：
    1. 防止梯度消失 或者 梯度爆炸
    2. 提高模型的训练速度 因为提高了收敛速度
    3. 打破对称性 让神经网络训练更强有力

参数初始化的方法：
    无法打破对称性的：
        全0， 全1，固定值
    打破对称性：
        随机初始化，正态分布初始化，kaiming初始化，xavier初始化

要使用ReLU及其系列：kaiming (深层网络)
要使用非ReLU及其系列：xavier (深层网络)
如果是浅层：优先考虑随机初始化

"""

#导包
import torch.nn as nn

#均匀分布随机初始化
def dm01():
    #创建一个线性全连接层 输入5个神经元 输出3个神经元
    linear = nn.Linear(5,3)
    #参2：下界 参3：上界
    nn.init.uniform_(linear.weight, -1, 1)
    nn.init.uniform_(linear.bias,-1,1)
    print(linear.weight.data)
    print(linear.bias.data)

#固定初始化
def dm02():
    #创建一个线性全连接层 输入5个神经元 输出3个神经元
    linear = nn.Linear(5,3)
    #参2：固定值
    nn.init.constant_(linear.weight, 10)
    nn.init.constant_(linear.bias, 10)
    print(linear.weight.data)
    print(linear.bias.data)

#全0初始化
def dm03():
    #创建一个线性全连接层 输入5个神经元 输出3个神经元
    linear = nn.Linear(5,3)
    #参2：全0初始化
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

#全1初始化
def dm04():
    #创建一个线性全连接层 输入5个神经元 输出3个神经元
    linear = nn.Linear(5,3)
    #参2：全1初始化
    nn.init.ones_(linear.weight)
    nn.init.ones_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

#正态分布随机初始化
def dm05():
    # 创建一个线性全连接层 输入5个神经元 输出3个神经元
    linear = nn.Linear(5, 3)
    # 参2：正态分布均值 参3：正态分布标准差
    nn.init.normal_(linear.weight, 0, 1)
    nn.init.normal_(linear.bias,0,1)
    print(linear.weight.data)
    print(linear.bias.data)

#kaiming初始化 kaiming/xavier初始化是给weight用的 bias不是二维是一维的
def dm06():
    # 创建一个线性全连接层 输入5个神经元 输出3个神经元
    linear = nn.Linear(5, 3)
    nn.init.kaiming_uniform_(linear.weight)
    print(f"kaiming均匀分布权重：{linear.weight}")

    nn.init.kaiming_normal_(linear.weight)
    print(f"kaiming正态分布权重：{linear.weight}")

#xavier初始化 kaiming/xavier初始化是给weight用的 bias不是二维是一维的
def dm07():
    # 创建一个线性全连接层 输入5个神经元 输出3个神经元
    linear = nn.Linear(5, 3)
    nn.init.xavier_uniform_(linear.weight)
    print(f"xavier均匀分布权重：{linear.weight}")

    nn.init.xavier_normal_(linear.weight)
    print(f"xavier正态分布权重：{linear.weight}")

if __name__ == '__main__':
    #dm01()      均匀分布随机初始化
    #dm02()      固定初始化
    #dm03()      全0初始化
    #dm06()
    dm07()
