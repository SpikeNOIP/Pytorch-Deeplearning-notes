"""
指数移动加权平均法:
    对一串随时间变化的数据做"平滑"，让每个新值都综合一
   点"过去的信息"，曲线更稳定、抖动更小。
   并不是一个梯度优化方法
"""

import torch
import matplotlib.pyplot as plt

ELEMENT_NUMBER = 30

def dm01():

     torch.manual_seed(0)
     temperature = torch.randn(size=[ELEMENT_NUMBER,])*10
     print(temperature)
     # 创建时间序列
     days = torch.arange(1,ELEMENT_NUMBER+1,1)
     plt.plot(days,temperature,color = "r")
     plt.show()

def dm02(beta = 0.9):

    torch.manual_seed(0)
    temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
    print(temperature)

    exp_weight_avg = []
    for idx, temp in enumerate(temperature,1):
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        new_temp = exp_weight_avg[idx - 2]*beta + (1-beta)*temp
        exp_weight_avg.append(new_temp)

    days = torch.arange(1,ELEMENT_NUMBER+1,1)
    plt.plot(days,exp_weight_avg,color = "r")
    plt.scatter(days,temperature,color = "b")
    plt.show()

if __name__ == '__main__':
    dm01()
    dm02()