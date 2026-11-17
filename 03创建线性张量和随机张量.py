"""
pytorch 里如何创建随机和线性张量
    线性张量：
    torch.arange() torch.linspace() 创建线性张量 注意和torch.range()有区别

    随机张量：
    torch.rand()/randn() 创建随机浮点类型张量
    torch.randint()
    torch.initial_seed() 和 torch.manual_seed() 随机种子设置
"""

import torch
from numpy.random.mtrand import randint


# torch.arange(start=, end=, step=): 创建指定步长的线性张量  左闭右开
# start: 起始值
# end: 结束值
# step: 步长, 默认1
# torch.linspace(start=, end=, steps=): 创建指定元素个数的线性张量  左闭右闭
# start: 起始值
# end: 结束值
# steps: 元素个数
# step=(end-start)/(steps-1)  value_i=start+step*i

def dm_01():
    data1 = torch.arange(start=0,end=10,step=2)     #左闭右开
    print(data1)

    # todo 创建一个等差张量 param1:startvalue param2:endvalue param3:elements counts
    data2 = torch.linspace(start=0,end=10,steps=3)
    print(data2)

def dm_02():
    t1 = torch.rand(size=(5,4)) #按照时间戳随机生成数据填充到指定维数张量中
    print(t1)

    # todo 设置随机种子 这个种子能够控制所有的随机张量生成
    # torch.initial_seed() 创建以时间戳为随机种子
    torch.manual_seed(seed=66)
    #机数生成器（RNG）设定一个起点。之后每次调用torch.rand()，都是从这条序列里依次取下一段随机数：

    #创建随机张量
    t2 = torch.rand(size=(5, 4))
    t3 = torch.rand(size=(5, 4))
    print(t2)
    print(t3)
    #所以 t2 和 t3 是序列里连续的两段，当然不一样 —— 这跟没设种子时一样。
    #结果
    #创建符合正态分布的随机张量
    t2 = torch.randn(size=(5, 4))
    print(t2)
    #创建整数张量
    t3 = torch.randint(0,10,size=(5,4))
    # torch.randint(low=, high=, size=): 创建指定形状指定范围随机整数类型张量  左闭右开
    # low: 最小值
    # high: 最大值
    # size: 形状, 元组
    print(t3)

if __name__ == "__main__":
    dm_02()

