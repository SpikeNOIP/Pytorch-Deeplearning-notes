"""
演示：
    演示二分类的损失函数
二分类任务的损失函数(BCELoss):
    公式：
        Loss = -ylog(预测值)-(1-y)log(1-预测值)
    细节：
        因为公式中没有包含Sigmoid激活函数，
        所以在使用BCELoss时，通常会和Sigmoid激活函数一起使用。
"""
import torch
import torch.nn as nn

def dm01():

    y_true = torch.tensor([1,0,0],dtype=torch.float32)

    y_pred = torch.tensor([0.9,0.2,0.1])

    criterion = nn.BCELoss()

    loss = criterion(y_pred,y_true)

    print(loss.detach().numpy())

if __name__ == '__main__':
    dm01()