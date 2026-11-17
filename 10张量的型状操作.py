"""
张量的形状操作

涉及到的API：
    reshape():      在不改变元素总个数的前提下，改变张量的形状
    unsqueeze():    在张量的指定位置插入一个维度
    squeeze():      移除张量中所有维度为1的维度
    permute():      一次可以同时交换多个维度
    transpose():    一次只能交换两个维度
    view():         与reshape()类似，但要求张量是连续的 连续张量= 内存中存储顺序 和 在张量中的显示顺序相同
    contiguous();   把不连续的张量 -> 连续的张量 基于张量中的显示顺序，修改内存中的存储顺序

"""
import torch
torch.manual_seed(24)


#reshape()函数：
def dm_01():
    t1 = torch.randint(1,10,size=(2,3))
    print(f"t1:{t1},shape:{t1.shape},row:{t1.shape[0]},columns:{t1.shape[1],t1.shape[-1]}")
    #shape[]这个操作就是把张量形状列表进行访问

    # t2 = t1.reshape(3, 2)
    # t2 = t1.reshape(6, 1)
    # t2 = t1.reshape(5, 1)  报错 因为原张量有6个元素，不能reshape为5行1列

#unsequeeze()函数 squeeze()函数
def dm_02():
    t1 = torch.randint(1,10,size=(2,3))
    print(f"t1:{t1},shape:{t1.shape}")
    # [
    #   [6, 9, 9]
    #   [2, 8, 7]
    # ]


    #在0维上加入一个维度
    t2 = t1.unsqueeze(0)
    print(f"t2:{t2},shape:{t2.shape}")
    # shape:[1, 2, 3]
    # [
    #   [
    #       [6, 9, 9]
    #       [2, 8, 7]
    #   ]
    # ]

    # 在1维上加入一个维度
    t3 = t1.unsqueeze(1)
    print(f"t3:{t3},shape:{t3.shape}")
    # shape:[2,1,3]
    # [
    #   [
    #     [ 6, 9, 9]
    #   ]
    #   [
    #     [2, 8, 7]
    #   ]
    # ]

    #在2维上加入一个维度
    t4 = t1.unsqueeze(2)
    print(f"t4:{t4},shape:{t4.shape}")
    # shape: [2,3,1]
    # [
    #   [[6], [9], [9]]
    #   [[2], [8], [7]]
    # ]

    #在3维度（不存在）上加入一个维度
    # t5 = t1.unsqueeze(3)
    # print(f"t5:{t5},shape:{t5.shape}") 报错

    #删除所有为1的维度
    t6 = torch.randint(1,10,size=(2,1,3,1,1))
    print(f"t6:{t6},shape:{t6.shape}")

    t7=t6.squeeze()
    print(f"t7:{t7},shape:{t7.shape}")

#permute()函数 transpose()函数
def dm_03():
    t1 = torch.randint(1,10,size=(2,3,4))

    #改变维度从(2,3,4)->(3,2,4)
    t2 = t1.transpose(0,1)  # 0和1维度交换 可以用-1
    print(f"t2:{t2},shape:{t2.shape}")
    #不会改变任何内容

    #改变维度从(2,3,4)->(4,2,3)
    t3 = t1.permute(2,0,1)  #可以一次性换多个维度
    print(f"t3:{t3},shape:{t3.shape}")
    # 不会改变任何内容

#view()函数 contiguous()函数 is_contiguous()函数
def dm_04():
    #view()无法改变不连续的张量形状。可以通过contiguous()，把不连续的张量 -> 连续的张量
    t1 = torch.randint(1,10,size=(2,3))
    #通过 t1.is_contiguous() 判断张量顺序和内存顺序是否一样 True

    t2 = t1.view(3,2)
    print(f"t2:{t2},shape:{t2.shape}")
    print(t2.is_contiguous()) #True

    #使用transpose()交换维度->交换之后 不连续了
    t3=t1.transpose(0,1)
    print(f"t3:{t3},shape:{t3.shape}")
    print(t3.is_contiguous()) #False

    #尝试把 t3张量 从(2,3) view(2,3)会报错

    #可以通过contiguous()，把不连续的张量 -> 连续的张量
    t4 = t3.contiguous().view(2,3)
    print(f"t4:{t4},shape:{t4.shape}")
    print(t4.is_contiguous()) #True

if __name__ == '__main__':
    # dm_02()
    # dm_03()
    dm_04()
