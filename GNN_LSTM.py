import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN_func import ConstraintGNN, calculate_accuracy, train_and_evaluate
from torch_geometric.nn import global_mean_pool
import torch_geometric.nn as geom_nn

class GNN_LSTM(nn.Module):
    def __init__(self, 
                 x_shape: int,
                 horizon_size: int,        
                 covariate_size: int,
                 hidden_size: int,
                 gnn_hidden: int,
                 layer_size: int,
                 target_width: int,  
                 device
                 ):
        super(GNN_LSTM, self).__init__()

        self.batch_size = 1
        
        self.horizon_size = horizon_size
        self.covariate_size = covariate_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.target_width = target_width
        self.x_shape = x_shape
        self.device = device
        

        self.input_size = covariate_size + target_width * (target_width - 1)

        self.conv1 = geom_nn.GCNConv(x_shape, gnn_hidden)
        self.conv2 = geom_nn.GCNConv(gnn_hidden, gnn_hidden)
        self.global_pool = global_mean_pool
        self.fc = nn.Linear(gnn_hidden * x_shape, self.target_width*(self.target_width-1))

        # Initialize weights for the gates
        self.W_ii = nn.Parameter(torch.Tensor(covariate_size + target_width * (target_width - 1), hidden_size))
        self.W_gi = nn.Parameter(torch.Tensor(target_width * (target_width - 1), hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_if = nn.Parameter(torch.Tensor(covariate_size + target_width * (target_width - 1), hidden_size))
        self.W_gf = nn.Parameter(torch.Tensor(target_width * (target_width - 1), hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.W_ig = nn.Parameter(torch.Tensor(covariate_size + target_width * (target_width - 1), hidden_size))
        self.W_gg = nn.Parameter(torch.Tensor(target_width * (target_width - 1), hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))

        self.W_io = nn.Parameter(torch.Tensor(covariate_size + target_width * (target_width - 1), hidden_size))
        self.W_go = nn.Parameter(torch.Tensor(target_width * (target_width - 1), hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))


        # Reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters with uniform distribution
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, GNN_dataset, states=None):
        seq_len, batch_size, _ = input.size()
        seq_len = min(seq_len,len(GNN_dataset))
        

        if states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=input.device)
            h_t = h_t.double()
            c_t = torch.zeros(batch_size, self.hidden_size, device=input.device)
            c_t = c_t.double()
        else:
            h_t, c_t = states

        outputs = []

        for t in range(seq_len):
            #print("现在运行到第",t,"轮")
            
            gnn_t = GNN_dataset[t]
            x_t = input[t]
            #print(gnn_t)
            #print("输入层维度为：",x_t.size())

            x, edge_index = gnn_t.x, gnn_t.edge_index
            
            x = torch.tensor(x)
            edge_index = torch.tensor(edge_index)
            
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            # print(x.type())
            # print(edge_index.type())
            
            x = x.to(torch.float64)
            edge_index = edge_index.to(torch.int64)
            
            # print(x.type())
            # print(edge_index.type())

            # 只对变量节点进行预测（bipartite=0的节点）
            var_mask = gnn_t.x[:, 0] == 1  # 第一列为1的是变量节点
            var_nodes = torch.where(var_mask)[0]

            # 仅用变量节点进行前向传播
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            x=x.view(1,-1)
            #print(x.shape)
            out = torch.relu(self.fc(x))
            gnn_t = out.view(self.batch_size, self.target_width * (self.target_width - 1))
            #print("gnn输出格式为：",out.size())

            # LSTM gates
            self.W_ii = self.W_ii.double()
            self.W_hi = self.W_hi.double()
            self.b_i = self.b_i.double()
            i_t = torch.sigmoid(torch.matmul(x_t, self.W_ii) + torch.matmul(gnn_t, self.W_gi) + torch.matmul(h_t, self.W_hi) + self.b_i)
            f_t = torch.sigmoid(torch.matmul(x_t, self.W_if) + torch.matmul(gnn_t, self.W_gf) + torch.matmul(h_t, self.W_hf) + self.b_f)
            g_t = torch.tanh(torch.matmul(x_t, self.W_ig) + torch.matmul(gnn_t, self.W_gg) + torch.matmul(h_t, self.W_hg) + self.b_g)
            o_t = torch.sigmoid(torch.matmul(x_t, self.W_io) + torch.matmul(gnn_t, self.W_go) + torch.matmul(h_t, self.W_ho) + self.b_o)

            # Update cell state
            c_t = f_t * c_t + i_t * g_t

            # Update hidden state
            h_t = o_t * torch.tanh(c_t)
            #print("隐藏层维度为：",h_t.size())

            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=0)
        return outputs, (h_t, c_t)

# # Example usage:
# input_size = 10
# hidden_size = 20
# seq_len = 7
# batch_size = 3

# model = CustomLSTM(input_size, hidden_size)
# inputs = torch.randn(seq_len, batch_size, input_size)
# outputs, (h_n, c_n) = model(inputs)

# print(outputs.shape)  # Output shape: (seq_len, batch_size, hidden_size)
# print(h_n.shape)      # Last hidden state shape: (batch_size, hidden_size)
# print(c_n.shape)      # Last cell state shape: (batch_size, hidden_size)