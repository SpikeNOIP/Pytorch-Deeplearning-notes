"""
张量的基本运算：
涉及到的API：
    加减乘除取负 tensor add/sub/mul/div/neg
    如果想要修改原数据，需要使用inplace操作，即在操作符后加_ add_/sub_/mul_/div_/neg_
"""

import torch

t1 = torch.tensor([1,2,3,4,5,6])
t2 = torch.tensor([1,2,3,4,5,6])
t3 = torch.tensor([1,2,3,4,5,6])
t4 = torch.tensor([1,2,3,4,5,6],dtype=torch.float64)
t5 = torch.tensor([1,2,3,4,5,6])

t6 = t1.add(10) # 等价于 t2 = t1 +10

t1.add_(10)
t2.sub_(10)
t3.mul_(10)
t4.div_(3)
t5.neg_()

print(t1)
print(t2)
print(t3)
print(t4)
print(t5)

