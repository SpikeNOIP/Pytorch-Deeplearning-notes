"""
一个张量一旦设置了 自动微分 这个张量就不能直接转成numpy的ndarray对象了
需要使用detach()
"""

import torch

t1 = torch.tensor([10,20],requires_grad=True,dtype=torch.float)
print(f"t1:{t1},type:{type(t1)}")

t2=t1.detach()

t1.data[0]=100
print(f"t1:{t1}")
print(f"t1:{t2}")
#共享一个内存

n1=t2.numpy()
print(f"t2:{t2}")

print(t1.requires_grad) #True 可以微分
print(t2.requires_grad) #False 不能微分

#最终
#n2 = t1.detach().numpy()