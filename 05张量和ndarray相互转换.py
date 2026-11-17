"""
演示张量和 numpy之间如何相互转换 以及如何从标量张量中提取内容

涉及到的API：
    场景1：张量 ——> numpy nd数组
        张量对象.numpy()            共享内存
        张量对象.numpy().copy()     不共享内存，链式编程
    场景2：numpy nd数组 ——> 张量
        from_numpy()               共享内存
        torch.tensor(nd数组)       不共享内存，链式编程
    场景3：从标量张量中 提取内容
        标量张量（只有一个数值的张量）.item()

    ndarray和tensor的区别：
     - 本质：都是多维数组。
     - ndarray = 纯计算数组（NumPy生态：科学计算、数据处理）。
     - Tensor = 带"深度学习超能力"的数组（GPU 加速 + 自动求导），所以深度学习的"张量"特指它。

"""
import torch
import numpy as np


#张量——>numpy nd数组
def dm_01():
    t1 = torch.tensor([1,2,3,4,5])
    #n1 = t1.numpy() 这个方法共享内存
    n1=t1.numpy().copy() #这个方法不共享内存 会分开

    n1[0]=100
    print(n1)
    print(t1)

def dm_02():
    n1 = np.array([1,2,3,4,5])

    # t1 = torch.from_numpy(n1) 这个方法要共享内存
    t1 = torch.from_numpy(n1).type(torch.int64)

    t1[0]=100
    print(t1)
    print(n1)

def dm_03():
    t1 = torch.tensor(100)
    print(t1.item())

    t2 = torch.tensor([100])       #能读出来
    print(t2.item())

    t3 = torch.tensor([1,2,3,4,5]) #.item()方法只适用于标量张量
    print(t3.item())

if __name__ == '__main__':
    # dm_01()
    # dm_02()
    dm_03()