"""
整个PyTorch模型构建的流程：
    1.准备数据集
    2.构建使用的模型
    3.创建损失函数和优化器
    4.模型训练
"""
#导入相关模块
import torch
from statsmodels.tools.eval_measures import bias
from torch.utils.data import TensorDataset      #构造数据集对象
from torch.utils.data import DataLoader         #数据加载集
import matplotlib.pyplot as plt                 #绘图
from torch import nn                            #nn模块中有平方损失函数和假设函数
from torch import optim                         #优化器
from sklearn.datasets import make_regression    #生成回归数据

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False
#这两行通常是成对出现的：第一行解决"中文"，第二行解决"中文带来的负号副作用"。所以凡是用
#matplotlib 画带中文的图，开头就加上这两行。

#numpy -> 张量 -> 数据集 -> 数据加载器
def crate_dataset():
    x , y, coef=make_regression(
        n_samples=100,          # 设置有多少数据 多少行
        n_features=1,           # 有多少特征 只有一个
        noise=10,               # 噪声
        coef=True,              # 是否返回系数
        bias=14.5,              # 偏置
        random_state=3,         # 随机种子
    )

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return x,y,coef

def train_model(x,y,coef):
    #1 准备数据集 准备数据加载器
    dataset = TensorDataset(x, y)
    #参数1：数据集对象 参数2：批次大小 参数3：是否打乱洗匀
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    #创建线性回归模型
    model = nn.Linear(1, 1)
    #创建损失函数
    criterion = nn.MSELoss()
    #创建优化器 参1 模型参数 参2 学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs, loss_list, total_loss, total_samples = 100, [], 0.0, 0
    #要注意这里和机器学习sklearn库的不同，sklearn 把训练"封装隐藏"了
    #这里需要自己写训练代码，包括前向传播，损失计算，反向传播，优化器更新
    #y_pred = model(train_x)这一步里面的w是在模型初始化时自带的 用的初始随机值
    for i in range(epochs):
        #每轮分批次训练 一共100轮 每轮一共有7个批次
        for train_x, train_y in dataloader:
            #前向传播
            y_pred = model(train_x)
            loss = criterion(y_pred, train_y.reshape(-1,1)) #这里的
            total_loss += loss.item()
            total_samples += 1
            #反向传播
            optimizer.zero_grad()   #清空梯度
            loss.sum().backward()   #整个 loss 对 w、b 的偏导数
            optimizer.step()        #用梯度下降更新 w、b 的值

        loss_list.append(total_loss / total_samples)
        print(f"迭代轮数：{i+1}, 平均损失值：{total_loss / total_samples}")

    print(f"经过100轮迭代，模型参数预测，权重：{model.weight},偏置：{model.bias}\n")
    print(f"模型真实参数，权重：{coef},偏置：14.5")
    #模型预测结果 w=27.5838 b=13.9248 y = 27.5838x + 13.9248

    #绘制损失曲线
    plt.plot(range(epochs), loss_list)
    plt.title("损失值曲线变化图")
    plt.grid()
    plt.show()

    #绘制散点图和预测线图
    plt.scatter(x,y) #散点图
    y_pred = torch.tensor(data=[ v*model.weight+model.bias for v in x ])
    y_true = torch.tensor(data=[ v*coef + 14.5 for v in x ])
    plt.plot(x, y_pred, color='red',label="预测曲线")
    plt.plot(x,y_true, color='green',label="真实曲线")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    x,y,coef = crate_dataset()
    train_model(x, y, coef)