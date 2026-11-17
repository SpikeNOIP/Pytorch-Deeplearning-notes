import torch

# torch.tensor(data=, dtype=):
# dtype: 指定元素类型, 浮点类型默认是float32

# tensor.type(dtype=): 修改张量元素类型
# torch.float32
# torch.FloatTensor
# torch.cuda.FloatTensor

data1 = torch.tensor(data=[1,2,3,4,5,6,7,8],dtype=torch.float)
print(f"data1:{data1}, type(data1):{data1.dtype}") #使用dtype去查看数值类型 type(data)会始终显示Tensor

#方法1 使用type转换
data2 = data1.type(torch.int64)
print(f"data2:{data2}, type(data2):{data2.dtype}")
data3 = data2.type(torch.int16)
print(f"data3:{data3}, type(data3):{data3.dtype}")

t4 = torch.tensor(data=[1,2,3,4,5,6,7,8])
t5 = t4.int()
print(t5)
