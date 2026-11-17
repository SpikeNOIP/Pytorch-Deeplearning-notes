"""
梯度下降：
    w新 = w旧 - 学习率 * 梯度
    梯度优化方法并不是修改这个公式
    而是 从是通过调整 学习率/梯度的算法 来优化w

存在问题：
    1.遇到平缓区域 梯度下降缓慢 权重更新会特别慢
    2.遇到鞍点（导数为0）处 无法更新
    3.可能会遇到局部最优解 但是不是最好的点
优化方法：
    1.动量法 Momentum 通过修正梯度来加速收敛
    2.自适应学习率 AdaGrad 通过修正学习率来加速收敛
    3.RMSProp 通过修正学习率来加速收敛
    4.Adam 通过综合结合修正学习率+梯度来加速收敛
    综合:Adam

    动量法：
        通过引入动量项 来加速收敛
        公式：
            St = β * St-1 + (1 - β) * Gt
                St: 本次的指数移动加权平均结果 ---> 丢进 w新 = w旧 - 学习率 * 梯度 更新权重
                β: 调节系数，数据越大越平缓
                Gt: 本次的梯度

    自适应学习率：AdaGrad
        通过引入历史梯度的平方和 来修正学习率
        公式：
            St = St-1 + Gt*Gt
                St: 累加历史梯度的平方和
                St-1: 历史累计平方梯度
                Gt: 本次的梯度
        学习率 = 学习率/ (sqrt(St) + 小常数)
        梯度下降；
            w新 = w旧 - 调整后的学习率 * 梯度
        问题：
            可能会使 学习率过度的下降 导致学习率太小难以找到最优解

    RMSProp: 可以看做是对 AdaGrad做的优化 加入调和权重系数
        公式：
            St = β * St-1 + (1 - β) * Gt*Gt
                St: 本次的指数移动加权平均结果
                β: 调节系数，数据越大越平缓
                Gt: 本次的梯度
        学习率：
            学习率 = 学习率/ (sqrt(St) + 小常数)
        梯度下降；
            w新 = w旧 - 调整后的学习率 * 梯度

    Adam: 综合了动量法和自适应学习率的优点，通过综合结合修正学习率+梯度来加速收敛
        公式：
            St = β1 * St-1 + (1 - β1) * Gt
                St: 本次的指数移动加权平均结果
                β1: 调节系数，数据越大越平缓
                Gt: 本次的梯度
            Rt = β2 * Rt-1 + (1 - β2) * Gt*Gt
                Rt: 本次的指数移动加权平均结果
                β2: 调节系数，数据越大越平缓
                Gt: 本次的梯度
        学习率：
            学习率 = 学习率/ (sqrt(Rt) + 小常数)
        梯度下降；
            w新 = w旧 - 调整后的学习率 * 梯度
"""
import torch
import torch.nn as nn
import torch.optim as optim

#1.Momentum优化方法 优化梯度
def dm01_Momentum():
    #初始化权重系数
    w = torch.tensor([1],requires_grad=True,dtype=torch.float32)
    #创建损失函数
    criterion = (w ** 2) / 2
    #创建优化器 传参时传入momentum参数 作为β
    #Momentum就是在SGD的基础上增加了一个动量项 传入momentum参数
    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)
    #进行优化
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"动量法第一次优化 w:{w},w_grad:{w.grad}")

    criterion = (w ** 2) / 2
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"动量法第二次优化 w:{w},w_grad:{w.grad}")

#2.AdaGrad优化方法 优化学习率
def dm02_AdaGrad():
    #初始化权重系数
    w = torch.tensor([1],requires_grad=True,dtype=torch.float32)
    #创建损失函数
    criterion = (w ** 2) / 2
    #创建Adagrad优化器
    optimizer = optim.Adagrad(params=[w], lr=0.01)
    #进行优化
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"AdaGrad第一次优化 w:{w},w_grad:{w.grad}")

    criterion = (w ** 2) / 2
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"AdaGrad第二次优化 w:{w},w_grad:{w.grad}")

#2.RMSProp优化方法 优化学习率
def dm03_RMSProp():
    #初始化权重系数
    w = torch.tensor([1],requires_grad=True,dtype=torch.float32)
    #创建损失函数
    criterion = (w ** 2) / 2
    #创建RMSprop优化器 传入alpha参数 作为β
    optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.98)
    #进行优化
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"RMSprop第一次优化 w:{w},w_grad:{w.grad}")

    criterion = (w ** 2) / 2
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"RMSprop第二次优化 w:{w},w_grad:{w.grad}")

#3.Adam自适应矩故居优化方法 优化学习率
def dm04_Adam():
    #初始化权重系数
    w = torch.tensor([1],requires_grad=True,dtype=torch.float32)
    #创建一个损失函数
    criterion = (w ** 2) / 2
    #创建Adam优化器 传入beta1 beta2参数 作为β1 β2
    optimizer = optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.999))
    #进行优化
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"Adam第一次优化 w:{w},w_grad:{w.grad}")

    criterion = (w ** 2) / 2
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"Adam第二次优化 w:{w},w_grad:{w.grad}")

if __name__ == '__main__':
    dm01_Momentum() #[0.9900] [0.9711]
    dm02_AdaGrad()  #[0.9900] [0.9830]
    dm03_RMSProp()  #[0.9293] [0.8809]
    dm04_Adam()     #[0.9900] [0.9800]