"""
手机价格预测案例
    使用ANN神经网络预测手机价格 /也可以用机器学习 但是特征工程很复杂 用深度学习更方便

    数据集：https://www.kaggle.com/iabhishek/price-range-prediction-dataset

    基于手机的20个特征 预测手机价格至四个区间

ANN神经网络的搭建流程：
    1.构建数据集
    2.搭建神经网络
    3.模型训练
    4.模型测试

"""
from enum import unique
from typing import Self

import torch                                # PyTorch 封装了张量的各种操作
from torch.utils.data import TensorDataset  # 数据集对象 数据-Tensor-数据集-数据加载器
from torch.utils.data import DataLoader     # 数据加载器
import torch.nn as nn                       # 神经网络 Neural-Network 封装了各种神经网络层
import torch.optim as optim                 # 优化器
from sklearn.model_selection import train_test_split    # 数据集分割
import pandas as pd                         # 数据处理
import numpy as np                          # 数值计算
import matplotlib.pyplot as plt             # 绘图
import time                                 # 时间
from torchsummary import summary            # 模型参数检查

#todo 1 数据集准备与构建
def create_dataset():
    #1 加载数据集
    data = pd.read_csv("mobile-price-range-predict.csv")
    #print(data.head())
    #print(data.shape) (2000, 21) 2000行 21列

    #2 提取特征/标签 feature/label
    x,y = data.iloc[:, :-1], data.iloc[:, -1]
    #print(f"x:{x}, y:{y}")
    #print(f"x.shape:{x.shape}, y.shape:{y.shape}") x.shape:(2000, 20), y.shape:(2000,)
    x=x.astype(np.float32)

    #3 数据集分割
    result = train_test_split(x,y,random_state=10,test_size=0.2,stratify=y)
    x_train = result[0]
    x_test = result[1]
    y_train = result[2]
    y_test = result[3]

    #4 封装成 张量数据集：数据->Tensor->张量数据集TensorDataset->数据加载器DataLoader
    train_data = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_data = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    #print(f"train_data:{train_data}, test_data:{test_data}") 返回对象值 train_data:<torch.utils.data.dataset.TensorDataset object at 0x000001BDD051F4D0>, test_data:<torch.utils.data.dataset.TensorDataset object at 0x000001BDD059E5D0>

    #5 返回结果 参1.训练数据集 参2.测试数据集 参3.特征数(输入层维度) 参4.类别数(输出层维度)
    return train_data, test_data, x_train.shape[1], len(np.unique(y))

#todo 2 搭建神经网络
class PriceANN(nn.Module):
    #1 在constructor里初始化父类成员
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #2 搭建神经网络
        #隐藏层1
        self.linear1 = nn.Linear(input_dim, 128)
        #隐藏层2
        self.linear2 = nn.Linear(128, 256)
        #输出层
        self.output = nn.Linear(256, output_dim)

    #3 前向传播 forward()
    def forward(self, x) -> torch.Tensor:
        #隐藏层1： 加权求和 + 激活函数(relu)
        x= torch.relu(self.linear1(x))
        #隐藏层2： 加权求和 + 激活函数(relu)
        x= torch.relu(self.linear2(x))
        #输出层： 加权求和 + 激活函数(softmax)
        #这里没有用 x = softmax(self.output(x)) 因为后面损失函数要用多分类交叉熵损失函数CrossEntropyLoss
        #CrossEntropyLoss已经包含了softmax激活函数
        x= self.output(x)
        return x

#todo 3 模型训练
#模型的训练是写在一个单独的函数里
#包含步骤 1.输入数据读取 2 送入网络 3计算损失 4更新参数 5模型保存
#只用传3个参数 训练数据集 输入维度 输出维度
def train_model(train_data, input_dim, output_dim):
    #固定随机种子
    torch.manual_seed(10)
    #创建数据加载器 DataLoader
    #参1：数据集  参2：批量大小  参3：是否打乱数据
    dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    #创建模型
    model = PriceANN(input_dim, output_dim)
    #定义损失函数
    criterion = nn.CrossEntropyLoss()
    #定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    #定义训练轮数
    epchos=50
    for epcho in range(epchos):
        #训练开始时间
        start_time = time.time()
        #计算损失值
        total_loss=0.0
        #计算在哪一批次
        batch_count=0
        #遍历数据加载器
        for x,y in dataloader:
            #设置模型为训练模式
            model.train()
            #前向传播
            y_pred = model(x)
            #计算损失
            loss=criterion(y_pred, y)
            #梯度清零 放在反向传播之前
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #更新参数
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
        print(f"Epoch {epcho+1}, Loss: {total_loss/batch_count}")

    print(f"\n\n 模型参数：{model.state_dict()} \n\n")
    #和机器学习不一样 机器学习保存的是模型 深度学习只保存模型参数 参数量通常比模型小很多
    torch.save(model.state_dict(), "model/price_model.pth")

#todo 4 模型测试
def evaluate(test_data, input_dim, output_dim):
    #创建神经网络
    model = PriceANN(input_dim, output_dim)
    #加载模型数据
    model.load_state_dict(torch.load("model/price_model.pth"))
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    #定义变量
    correct = 0
    #从数据集中加载数据，获得每批次的数据
    for x,y in test_loader:
        #设置为评估模式
        model.eval()
        #模型预测
        y_pred= model(x)
        print(f"y_pred:{y_pred}, y:{y}")
        #[[概率，概率],[],[]...]
        

if __name__ == '__main__':
    train_data,test_data,input_dim,output_dim=create_dataset()
#print(input_dim,output_dim) 20 4 20个特征 4个输出结果
#    price_model = PriceANN(input_dim,output_dim)
#    summary(price_model,input_size=(16,input_dim))
# ----------------------------------------------------------------
#          Layer (type)               Output Shape         Param #
# ================================================================
#             Linear-1              [-1, 16, 128]           2,688
#             Linear-2              [-1, 16, 256]          33,024
#             Linear-3                [-1, 16, 4]           1,028
# ================================================================
#     train_model(train_data, input_dim, output_dim)
    evaluate(test_data, input_dim, output_dim)
