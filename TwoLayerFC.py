#%%
import torch
import torch.nn as nn

class TwoLayerFC(nn.Module):   #定义了一个两层全连接神经网络类 TwoLayerFC，继承自 PyTorch 的 nn.Module
      
    #搭建网络框架
    def __init__(self, input_size, hidden_size, output_size, device):
        super(TwoLayerFC, self).__init__()
        # 第一层全连接层 fc1；
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 激活函数（这里使用ReLU）
        self.relu = nn.ReLU()
        # 第二层全连接层
        self.fc2 = nn.Linear(hidden_size, output_size)
    


        # TwoLayerFC 类中定义的 forward 方法，描述了数据在神经网络中的前向传播过程
    def forward(self, x):
        # 前向传播
        out = self.fc1(x)   #将输入 x 通过第一层全连接层 fc1 进行前向传播，得到输出 out
        out = self.relu(out) #将输出 out 通过 ReLU 激活函数进行非线性转换；
        out = self.fc2(out) #将激活后的输出 out 通过第二层全连接层 fc2 进行前向传播，得到最终的输出
        return out