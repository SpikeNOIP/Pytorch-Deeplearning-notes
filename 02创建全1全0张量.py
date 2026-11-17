"""
torch.ones(size=(3,4)) 根据形状创建全1张量
torch.ones_like(input=) 根据指定形状创建指定张量
"""

import torch

def dm_01():
    data1 = torch.ones(size=(3,4))
    print(data1)
    print(data1.dtype)
    print(data1.shape)

    data2 = torch.tensor([1,2,3,4,5])
    data2 = torch.ones_like(data2)
    print(data2)
    print(data2.dtype)
    print(data2.shape)

def dm_02():
    data1 = torch.zeros(size=(3,4))
    print(data1)
    print(data1.dtype)
    print(data1.shape)

    data2 = torch.tensor([1,2,3,4,5])
    data2 = torch.zeros_like(data2)
    print(data2)
    print(data2.dtype)
    print(data2.shape)

def dm_03():
    data1 = torch.zeros(size=(3,4))
    print(data1)
    print(data1.dtype)
    print(data1.shape)

    data2 = torch.tensor([1,2,3,4,5])
    data2 = torch.zeros_like(data2)
    print(data2)
    print(data2.dtype)
    print(data2.shape)

# torch.full(size=, fill_value=): 根据形状和指定值创建指定值的张量
# torch.full_like(input=, fill_value=): 根据指定张量形状和指定值创建指定值的张量

def dm_03():
	t1 = torch.full(size=(2, 3, 4), fill_value=255)
	t2 = torch.tensor(data=[[1, 2], [3, 4]])
	t3 = torch.full_like(input=t2, fill_value=255)
	print('t1的值是->', t1)
	print('t1的形状是->', t1.shape)
	print('t3的值是->', t3)
	print('t3的形状是->', t3.shape)



if __name__ == "__main__":
    dm_01()
    dm_02()
    dm_03()