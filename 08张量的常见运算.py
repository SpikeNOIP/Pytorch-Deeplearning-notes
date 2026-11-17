"""
演示张量常用运算
涉及到的API：
    sum() max() min() mean()   ——> 有dim参数
    sqrt() pow() log() log2() log10() ——> 没有dim参数
"""
import torch

t1 = torch.tensor([
    [1,2,3],
    [4,5,6]],
    dtype=torch.float
)
print(t1)

#sum()演示
print(t1.sum(dim=0)) #按列求和
print(t1.sum(dim=1)) #按行求和
print(t1.sum())      #整体求和

print("="*30)
print(t1.max(dim=0))    #按列求最大值
print(t1.max(dim=1))    #按行求最大值
print(t1.max())         #整体求最大值

print("="*30)
print(t1.mean(dim=0))   #按列求平均值
print(t1.mean(dim=1))   #按行求平均值
print(t1.mean())        #整体求平均值

print("="*30)
print(t1.sqrt())

