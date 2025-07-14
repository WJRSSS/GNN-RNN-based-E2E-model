####
#算法1


import numpy as np
import torch 

# %%

#projection1为改配约束
def projection1(A,B,w):
    # A的格式为: [target_width,target_width * 2]
    # B的格式为: [target_width,]
    # w的格式为: (seq_len, batch_size, horizon_size, quantile_size, target_width)
    #print("更改前:",w)
    #以下部分为添加松弛变量的改配约束
    # for s in range(w.shape[0]):
    #     for b in range(w.shape[1]):
    #         for h in range(w.shape[2]):
    #             for q in range(w.shape[3]):
    #                 AA_T = torch.mm(A, A.t())    #计算 A*A_T
    #                 inverse_AA_T = torch.inverse(AA_T)   #计算 (AA_T)^{-1}；torch.inverse 是 PyTorch 中计算矩阵逆的函数
                    
    #                 residual = torch.mm(A,w[s,b,h,q].view(-1, 1))   #计算 A*w
                    
    #                 residual = torch.sub(residual,B.view(-1, 1))    #计算 A*w - b
    #                 temp = torch.mm(inverse_AA_T, residual)     #计算(AA_T)^{-1}* (A*w - b)
    #                 w[s,b,h,q] = torch.sub(w[s,b,h,q], torch.mm(A.t(), temp).view(-1))
    #print("更改后:",w)

    #以下部分为使用relu函数的改配约束
    new_w = w.clone()
    new_w = w - B
    new_w = -new_w
    new_w_1 = torch.relu(new_w).clone() 
    new_w_1 = -new_w_1 + B
    #print(new_w_1)
    return new_w_1



# %%

# def projection2(w,Num_of_Regions,len_w):   
def projection2(w):   
    #确定需要应用 ReLU 的位置
    # positions_to_apply_relu = list(range(Num_of_Regions**2 + Num_of_Regions**2 + Num_of_Regions)) +\
    #       list(range(3*Num_of_Regions**2 + Num_of_Regions,len_w))
    new_w = w.clone()   #创建一个新的向量 new_w，用于存储经过投影后的结果
    # new_w[positions_to_apply_relu] = torch.relu(w[positions_to_apply_relu])  #应用ReLU操作，将输入的负值截断为 0，而保持非负值不变
    new_w = torch.relu(w).clone() 
    return new_w


# %%
#定义投影函数（ projection_k）
#作用：执行 k 次循环投影，结合 projection1 和 projection2 的操作。
# 关系：在模型训练的每一步迭代中，使用这个函数将模型输出的 w 投影到满足所有约束条件的空间中。
# 代码中使用这个函数进行多次投影，确保决策变量 w 满足线性约束和非负性约束

def projection_k(A,B, w, k = 8):
    '''
    做k次循环投影
    '''
    new_w = w.clone()
    for _ in range(k):
        new_w = projection1(A,B,new_w).clone()
        new_w = projection2(new_w).clone()
    #print(new_w)
    return new_w



# %%
#决策变量提取函数
#将决策向量 w 转换为具体的决策变量 e,f,a,v,y
def w_to_decisions(w,Num_of_Regions):
    e = np.zeros((Num_of_Regions,Num_of_Regions))
    f = np.zeros((Num_of_Regions,Num_of_Regions))
    a = np.zeros(Num_of_Regions)
    v = np.zeros((Num_of_Regions,Num_of_Regions))
    y = np.zeros(Num_of_Regions**2 - Num_of_Regions + 2*Num_of_Regions + Num_of_Regions**2 + Num_of_Regions**2 + Num_of_Regions)
    for i in range(Num_of_Regions):
        for j in range(Num_of_Regions):
            e[i][j] = w[Num_of_Regions*i + j]
    for i in range(Num_of_Regions):
        for j in range(Num_of_Regions):
            f[i][j] = w[Num_of_Regions**2 + Num_of_Regions*i + j]
    for i in range(Num_of_Regions):
        a[i] = w[2*Num_of_Regions**2 + i]
    for i in range(Num_of_Regions):
        for j in range(Num_of_Regions):
            v[i][j] = w[2*Num_of_Regions**2 + Num_of_Regions + Num_of_Regions*i + j]

    y = w[3*Num_of_Regions**2 + Num_of_Regions:].detach().numpy()

    return e,f,a,v,y




