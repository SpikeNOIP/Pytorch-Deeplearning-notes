"""
案例：
    演示张量的点乘 和 矩阵乘法操作
点乘：
    要求： 两个张量的维度保持一致，对应元素直接做 相应的操作
    API：
        t1*t2
        t1.mul(t2)
矩阵乘法：
    要求： 两个张量的维度满足矩阵乘法的要求 mn np ——> mp
    API：
        t1@t2
        t1.matmul(t2)
"""

import torch
import numpy as np

n1 = torch.tensor([[1,2,3],[4,5,6]])
n2 = torch.tensor([[1,2,3],[4,5,6]])
n3 = torch.tensor([[1,2],[3,4],[5,6]])

print(n1*n2) #点乘
print(n1@n3) #矩阵乘法

