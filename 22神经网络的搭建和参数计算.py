"""
神经网络的搭建流程
四个步骤：
    1.准备数据集
    2.搭建神经网络（前向传播）
    3.模型训练 （反向传播）
    4.模型测试
对比机器学习模型搭建：1.数据集 2.数据预处理 3.特征工程 4.模型搭建 5.模型训练 6.模型评估

神经网络搭建流程：
    1.定义一个类，继承：nn.Module
    2.在 __init__() construction里搭建神经网络 多少层 多少个神经元 参数初始化方法/激活函数
    3.写 forward() function 前向传播
"""

import torch
import torch.nn as nn
from torchsummary import summary

#todo:2 搭建神经网络（继承nn.Module/forward()前向传播方法）
class Net(nn.Module):
    #todo: 2.1 construction 继承/定义神经网络架构/初始化方法
    def __init__(self):
        #初始化父类的构造函数 继承字段
        super().__init__()

        #搭建神经网络 隐藏层+输出层(不包含输入层)
        #隐藏层1：3个输入 3个输出
        self.linear1 = nn.Linear(3,3)
        #隐藏层2：3个输入 2个输出
        self.linear2 = nn.Linear(3,2)
        #输出层：命名output 2个输入 2个输出
        self.output = nn.Linear(2,2)

        #参数初始化
        #隐藏层1：
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        # 隐藏层2：
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    #todo:2.2 前向传播
    def forward(self,x):
        #计算第一个隐藏层的预测值 self.linear1(x)：线性求和再套sigmoid激活得到预测
        x=torch.sigmoid(self.linear1(x))
        # 计算第二个隐藏层的预测值 self.linear2(x)：线性求和再套relu激活得到预测
        x=torch.relu(self.linear2(x))
        # 计算输出层的预测值 self.output(x)：线性求和再套softmax激活得到预测
        x=torch.softmax(self.output(x),dim=-1)
        #返回预测值x
        return x

#todo:3 模型训练 和神经网络搭建平级
def train():

    #准备数据集
    torch.manual_seed(10)
    data = torch.randn(size=(5,3))  # 5个样本 每个样本3个特征 5行3列
    print(f"数据集为：{data}")

    #加载模型
    model = Net()
    print(f"模型为：{model}")
    # 模型为：Net(
    # (linear1): Linear(in_features=3, out_features=3, bias=True)
    # (linear2): Linear(in_features=3, out_features=2, bias=True)
    # (output) : Linear(in_features=2, out_features=2, bias=True)
    # )

    #调用神经网络进行训练 会自动调用forward前向传播
    output = model(data)

    print(f"预测结果：{output}")
    print("="*30,"计算模型参数","="*30)

    summary(model,input_size=(5,3))
    #隐藏层1：12个参数 隐藏层2：8个参数 输出层：6个参数

    for names,params in model.named_parameters():
        print(f"name:{names}")
        print(f"params:{params}\n")

if __name__ == '__main__':
    train()