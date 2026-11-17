"""
涉及到的API:
    cat():          沿着指定的维度拼接 除了拼接的那个维度外 其他维度数必须保持一致
    stack():        沿着一个新的维度拼接 所有的维度必须保持一致
    cat()和stack()的区别：cat()沿着指定的维度拼接 除了拼接的那个维度外 其他维度数必须保持一致 stack()沿着一个新的维度拼接 所有的维度必须保持一致

"""
import torch

torch.manual_seed(24)

t1=torch.randint(1,10,size=(2,3))
print(f"t1:{t1},shape:{t1.shape}")
# [[6, 9, 9],
#  [2, 8, 7]]

t2=torch.randint(1,10,size=(4,3))
print(f"t2:{t2},shape:{t2.shape}")
# [[8, 5, 8],
#  [4, 7, 4],
#  [3, 9, 3],
#  [6, 1, 4]]

t3=torch.randint(1,10,size=(2,4))
print(f"t3:{t3},shape:{t3.shape}")
# [[2, 8, 1, 2],
#  [5, 7, 4, 3]]


#todo 张量拼接
#使用cat()来拼接
t4=torch.cat((t1,t2),dim=0)
print(f"t4:{t4},shape:{t4.shape}")
# [[6, 9, 9],
#  [2, 8, 7],
#  [8, 5, 8],
#  [4, 7, 4],
#  [3, 9, 3],
#  [6, 1, 4]]

t5=torch.cat((t1,t3),dim=1)
print(f"t5:{t5},shape:{t5.shape}")
# [[6, 9, 9, 2, 8, 1, 2],
#  [2, 8, 7, 5, 7, 4, 3]]

#使用stack()来拼接
t6=torch.randint(1,10,size=(2,3))
t7=torch.stack([t1,t6],dim=0)
print(f"t7:{t7},shape:{t7.shape}")
# 按照0维来拼
# [[[6, 9, 9],
#   [2, 8, 7]],
#  [[4, 6, 5],
#   [8, 8, 8]]]

t8=torch.stack([t1,t6],dim=1)
print(f"t8:{t8},shape:{t8.shape}")
#按照1维度来拼
# [[[6, 9, 9],
#   [4, 6, 5]],
#  [[2, 8, 7],
#   [8, 8, 8]]]
t9=torch.stack([t1,t6],dim=2)
print(f"t9:{t9},shape:{t9.shape}")
# [[[6, 4],
#   [9, 6],
#   [9, 5]],
#  [[2, 8],
#   [8, 8],
#   [7, 8]]]

