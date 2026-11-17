""""
 张量（Tensor）是一个多维数组
    本质上就是多维数组在计算机里面的称呼
    深度学习里面：模型权重、输入数据、输出结果、梯度——全都是张量。
"""
import torch
import numpy as np

def dm01():
    #1. 标量 0阶张量 就是只有1个值的张量
    t1=torch.tensor(10)
    print(f"t1:{t1},type:{type(t1)}")

    #2. 二维数组 2阶张量 就是二维数组
    data=[[1,2,3],[4,5,6]]  #套了几个中括号就是几维数组
    t2=torch.tensor(data,dtype=torch.float)
    print(f"t2:{t2},type:{type(t2)}")

    #3. numpy nd数组 转成 张量
    #参1 下限（包含） 参2 上限（不包含） 参3（生成数组的形状）
    data=np.random.randint(0,10,size=(3,4))
    # print(f"data:{data},type:{type(data)}")
    t3=torch.tensor(data,dtype=torch.float)
    print(f"t3:{t3},type:{type(t3)}")

if __name__ == '__main__':
    dm01()