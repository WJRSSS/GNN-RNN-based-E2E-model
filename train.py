import torch
from Encoder import Encoder
from GNN_LSTM import GNN_LSTM
from Decoder import GlobalDecoder, LocalDecoder
from data import MQRNN_dataset
from torch.utils.data import DataLoader
from projection import *
from GNN_func import ConstraintGNN, calculate_accuracy, train_and_evaluate
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch,Data
from Demand_Forecast_FC import DemandForecastFC


def calc_loss(cur_series_covariate_tensor: torch.Tensor,
              next_covariate_tensor: torch.Tensor,
              cur_real_vals_tensor: torch.Tensor,
              cur_demand_tensor: torch.Tensor,
              cur_supply_tensor: torch.Tensor,
              cur_inventory_tensor: torch.Tensor,
              encoder: Encoder,
              gnn_lstm: GNN_LSTM,
              demand_forecast: DemandForecastFC,
              gdecoder: GlobalDecoder,
              ldecoder: LocalDecoder,
              cGNN: ConstraintGNN,
              batch_size: int,
              target_width:int,
              A_tensor,
              B_tensor,
              GNN_dataset:list,
              price,
              device):

    cur_series_covariate_tensor = cur_series_covariate_tensor.double()  # [batch_size, seq_len, target_width + covariate_size]
    next_covariate_tensor = next_covariate_tensor.double()  # [batch_size, seq_len, covariate_size * horizon_size]
    cur_real_vals_tensor = cur_real_vals_tensor.double()  # [batch_size, seq_len, horizon_size, target_width]
    cur_demand_tensor = cur_demand_tensor.double() # [batch_size, seq_len, horizon_size, target_width]
    cur_supply_tensor = cur_supply_tensor.double()
    cur_inventory_tensor = cur_inventory_tensor.double()
    #print(cur_series_covariate_tensor.size())
    print("需求矩阵宽度为：",cur_demand_tensor.size())
    #print(next_covariate_tensor.size())

    cur_series_covariate_tensor = cur_series_covariate_tensor.to(device)
    next_covariate_tensor = next_covariate_tensor.to(device)
    cur_real_vals_tensor = cur_real_vals_tensor.to(device)
    cur_demand_tensor = cur_demand_tensor.to(device)
    cur_supply_tensor = cur_supply_tensor.to(device)
    cur_inventory_tensor = cur_inventory_tensor.to(device)
    A_tensor = A_tensor.to(device)
    B_tensor = B_tensor.to(device)


    encoder.to(device)
    gnn_lstm.to(device)
    demand_forecast.to(device)
    gdecoder.to(device)
    ldecoder.to(device)

    # 转换形状
    cur_series_covariate_tensor = cur_series_covariate_tensor.permute(1, 0, 2)  # [seq_len, batch_size, target_width + covariate_size]
    next_covariate_tensor = next_covariate_tensor.permute(1, 0, 2)  # [seq_len, batch_size, covariate_size * horizon_size]
    cur_real_vals_tensor = cur_real_vals_tensor.permute(1, 0, 2, 3)  # [seq_len, batch_size, horizon_size, target_width]
    cur_demand_tensor = cur_demand_tensor.permute(1, 0, 2, 3) # [seq_len, batch_size, horizon_size, target_width]
    print("cur_demand_tensor.size",cur_demand_tensor.size())
    cur_supply_tensor = cur_supply_tensor.permute(1, 0, 2, 3)
    cur_inventory_tensor = cur_inventory_tensor.permute(1, 0, 2, 3)


    
    # [seq_len,batch_size,quantiles_size]
    LSTM_output = encoder(cur_series_covariate_tensor)
    print("GNN输入层格式:",len(GNN_dataset))
    print("原LSTM_Encoder输出层格式:",LSTM_output.size())
    GNN_LSTM_output, _ = gnn_lstm(cur_series_covariate_tensor,GNN_dataset)
    print("GNN_LSTM_Encoder输出层格式:",GNN_LSTM_output.size())
    # GNN_hidden = cGNN(GNN_dataset)
    # print("GNN输出层格式:",GNN_hidden.size())
    # GNN_hidden = GNN_hidden.to('cuda')
    # hidden_and_covariate = torch.cat([LSTM_output, GNN_hidden, next_covariate_tensor], dim=2)
    hidden_and_covariate = torch.cat([GNN_LSTM_output, next_covariate_tensor], dim=2)
    Gdecoder_output = gdecoder(hidden_and_covariate)
    context_size = ldecoder.context_size
    quantile_size = ldecoder.quantile_size
    horizon_size = encoder.horizon_size
    covariate_size = gdecoder.covariate_size
    seq_len = Gdecoder_output.shape[0]
    Demand_output = demand_forecast(hidden_and_covariate)
    print("demand输出格式为：",Demand_output.size())
    Demand_output = Demand_output.view(seq_len, batch_size, horizon_size, quantile_size, 1)
    print("demand输出格式为2：",Demand_output.size())
    print(Demand_output.size())
    # print(f"Gdecoder_output.shape: {Gdecoder_output.shape}")
    Gdecoder_output = Gdecoder_output.view(seq_len, batch_size, horizon_size + 1, context_size)
    horizon_agnostic_context = Gdecoder_output[:, :, -1, :]
    horizon_specific_context = Gdecoder_output[:, :, :-1, :]
    horizon_agnostic_context = horizon_agnostic_context.repeat(1, 1, horizon_size, 1)
    next_covariate_tensor = next_covariate_tensor.view(seq_len, batch_size, horizon_size, covariate_size)
    Ldecoder_input = torch.cat([horizon_specific_context, next_covariate_tensor], dim=3)
    # print(f"horizon_agnostic_context.shape: {horizon_agnostic_context.shape}")
    # print(f"Ldecoder_input.shape: {Ldecoder_input.shape}")
    horizon_agnostic_context = horizon_agnostic_context.permute(1, 0, 2, 3)
    Ldecoder_input = torch.cat([horizon_agnostic_context, Ldecoder_input],
                               dim=3)  # [seq_len, batch_size, horizon_size, 2*context_size+covariate_size]
    Ldecoder_output = ldecoder(Ldecoder_input)
    
    #由于包括松弛变量，输出值最后一个维度是target_width * 2，前半为决策变量，后半为松弛变量（2025/2/11 由于使用relu函数，最后一个维度依然是target_width）
    
    # 含qs：
    Ldecoder_output = Ldecoder_output.view(seq_len, batch_size, horizon_size, quantile_size, target_width * (target_width - 1))
    #Ldecoder_output = Ldecoder_output.view(seq_len, batch_size, horizon_size, target_width * (target_width - 1))
    
    #进行投影
    Ldecoder_output_PJ = projection_k(A_tensor,B_tensor, Ldecoder_output)
    print("Ldecoder_output_PJ.size",Ldecoder_output_PJ.size())
    #print(Ldecoder_output_PJ[0,0,0,0])

    #排除松弛变量，计算损失函数
    #Ldecoder_output_PJ = Ldecoder_output_PJ[:,:,:,:,:target_width]
    #print(Ldecoder_output_PJ)
    #Ldecoder_output_PJ = Ldecoder_output
    #print(Ldecoder_output_PJ)
    Ldecoder_output_net_income = Ldecoder_output_PJ.clone()
    cost_summary = Ldecoder_output_PJ.clone()
    Ldecoder_output_net_income = Ldecoder_output_net_income[...,:target_width]
    cost_summary = Ldecoder_output_net_income[...,:target_width]
    #不含qs，降1维
    for i in range(target_width):
        #Ldecoder_output_net_income数组存放每一个商品的净改配流入数量
        Ldecoder_output_net_income[:,:,:,:,i] = 0
        cost_summary[:,:,:,:,i] = 0
    #print(Ldecoder_output_net_income)
    #print(Ldecoder_output_net_income.shape)

    #print(Ldecoder_output_PJ)
    
    transformation_cost = 0.2
    #净改配流入
    for i in range(target_width):
        inc = 0
        out = 0
        for j in range(target_width):
            if j < i:
                inc += Ldecoder_output_PJ[:,:,:,:,j * (target_width - 1) + i - 1] 
            if j == i:
                continue
            if j > i:
                inc += Ldecoder_output_PJ[:,:,:,:,j * (target_width - 1) + i] 
        #inc = inc - Ldecoder_output_PJ[:,:,:,:,i * (target_width - 1) + i]
        for k in range(target_width - 1):
            if k < i:  
                out += Ldecoder_output_PJ[:,:,:,:,i * (target_width - 1) + k] * abs(price[i]-price[k]) * transformation_cost
            if k >= i:
                out += Ldecoder_output_PJ[:,:,:,:,i * (target_width - 1) + k] * abs(price[i]-price[k + 1]) * transformation_cost
        
        Ldecoder_output_net_income[:,:,:,:,i] = inc - out
        cost_summary[:,:,:,:,i] = out

        
        # Ldecoder_output_net_income[:,:,:,:,0] = Ldecoder_output_PJ[:,:,:,:,1] - Ldecoder_output_PJ[:,:,:,:,0]
        #print(Ldecoder_output_net_income)
        # Ldecoder_output_net_income[:,:,:,:,1] = Ldecoder_output_PJ[:,:,:,:,0] - Ldecoder_output_PJ[:,:,:,:,1]
    #print(Ldecoder_output_net_income)
    


    total_loss = torch.tensor([0.0], device=device)
    #print(cur_supply_tensor)

    real_demand = torch.sum(cur_demand_tensor,dim = -1, keepdim =True)
    #含qs：
    for i in range(quantile_size):
        is_modified = (torch.abs(Ldecoder_output_net_income[:,:,:,i,:]) > 0.01).float()
        base_reward = 10 * is_modified
        p = ldecoder.quantiles[i]
        #计算供给：总供给=期初库存+当期供给+净改配流入
        supply = cur_supply_tensor + Ldecoder_output_net_income[:,:,:,i,:] + cur_inventory_tensor
        #print(Ldecoder_output_net_income[:,:,:,i,:])
        #print(supply)
        #print(cur_demand_tensor)
        #取supply和demand的最小值
        #sales = torch.clamp(supply, max = cur_demand_tensor)
        supply = torch.clamp(supply, min=0)
        sales = torch.min(supply,cur_demand_tensor)
        #print(sales)
        for j in range(target_width):
            sales[:,:,:,j] = sales[:,:,:,j] * price[j]
        cost = cost_summary[:,:,:,i,:] 
        #print(cost)
        #print("收入",sales)
        #print("支出",cost)
        profit = sales - cost
        #print('利润为',profit)
    
        #误差=利润离500的距离+非负惩罚项SS
        #errors = abs(profit - 200) #+ 100000000 * torch.clamp(-Ldecoder_output_PJ[:,:,:,i,:], min = 0)
        real_demand
        demand_error = torch.abs(Demand_output[:,:,:,i,:] - real_demand)/50
        
        errors = 1 / (1 + torch.exp(profit/100000)) + demand_error - base_reward
        #print("利润误差为:",torch.sum(1 / (1 + torch.exp(profit/10))))
        #print("需求误差为:",torch.sum(demand_error))
    
        # cur_loss = torch.max((p-1)*errors,0)+torch.max(p*errors,0)#torch.max会把0认为是指定的维度，所以会返回一个元组
        cur_loss = torch.clamp((p - 1) * errors, min=0) + torch.clamp(p * errors, min=0)  
        # 使用clamp函数限制tensor的取值范围，代替了torch.max
        total_loss += torch.sum(torch.sum(errors))
    #print(total_loss)
    
    return total_loss



def custom_collate_fn(batch):
    # 初始化存储不同类型数据的列表
    cur_series_tensors = []
    cur_covariate_tensors = []
    cur_real_vals_tensors = []
    cur_demand_tensors = []
    cur_supply_tensors = []
    cur_inventory_tensors = []
    gnn_datasets = []

    # 遍历批次中的每个数据项
    sum = 0
    for item in batch:
        cur_series_tensors.append(item[0])
        cur_covariate_tensors.append(item[1])
        cur_real_vals_tensors.append(item[2])
        cur_demand_tensors.append(item[3])
        cur_supply_tensors.append(item[4])
        cur_inventory_tensors.append(item[5])
        gnn_datasets.append(item[6])
        # if sum < 1:
        #     gnn_datasets.extend(item[6])
        # sum += 1
        

    # 使用 torch.stack 处理张量数据
    cur_series_tensors = torch.stack(cur_series_tensors, dim=0)
    cur_covariate_tensors = torch.stack(cur_covariate_tensors, dim=0)
    cur_real_vals_tensors = torch.stack(cur_real_vals_tensors, dim=0)
    cur_demand_tensors = torch.stack(cur_demand_tensors, dim=0)
    cur_supply_tensors = torch.stack(cur_supply_tensors, dim=0)
    cur_inventory_tensors = torch.stack(cur_inventory_tensors, dim=0)
    #gnn_datasets = torch.stack(gnn_datasets, dim=0)

    # for data in gnn_datasets:
    #     if not isinstance(data, Data):
    #         raise TypeError(f"Expected torch_geometric.data.Data but got {type(data)}")

    # 使用 Batch.from_data_list 处理图数据
    #print("Batch前gnn_datasets为:",gnn_datasets)
    # gnn_datasets = Batch.from_data_list(gnn_datasets)
    #print("gnn_datasets为:",gnn_datasets)

    return (cur_series_tensors, cur_covariate_tensors, cur_real_vals_tensors, cur_demand_tensors, cur_supply_tensors, cur_inventory_tensors, gnn_datasets)



def train(encoder: Encoder,
          gnn_lstm: GNN_LSTM,
          demand_forecast: DemandForecastFC,
             gdecoder: GlobalDecoder,
             ldecoder: LocalDecoder,
             cGNN: ConstraintGNN,
             dataset: MQRNN_dataset,
             GNN_dataset:list,
             lr: float,
             batch_size: int,
             num_epochs: int,
             target_width:int,
             A_tensor,
             B_tensor,
             price,
             device):
     # 定义优化器
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    gnn_lstm_optimizer = torch.optim.Adam(gnn_lstm.parameters(), lr=lr)
    demand_forecast_optimizer = torch.optim.Adam(demand_forecast.parameters(), lr=lr)
    gdecoder_optimizer = torch.optim.Adam(gdecoder.parameters(), lr=lr)
    ldecoder_optimizer = torch.optim.Adam(ldecoder.parameters(), lr=lr)
    cGNN_optimizer = torch.optim.Adam(cGNN.parameters(), lr=lr)
    # 定义学习率调度器
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, 25, last_epoch=-1)
    scheduler1_1 = torch.optim.lr_scheduler.CosineAnnealingLR(gnn_lstm_optimizer, 25, last_epoch=-1)
    scheduler1_2 = torch.optim.lr_scheduler.CosineAnnealingLR(demand_forecast_optimizer, 25, last_epoch=-1)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(gdecoder_optimizer, 25, last_epoch=-1)
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(ldecoder_optimizer, 25, last_epoch=-1)
    scheduler4 = torch.optim.lr_scheduler.CosineAnnealingLR(cGNN_optimizer, 25, last_epoch=-1)

    # 数据加载器
    data_iter = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=custom_collate_fn)
    l_sum = 0.0

    
    #L1范数惩罚项 λ
    lambda_l1 = 100
    for i in range(num_epochs):
        print(f"epoch_num:{i}")
        epoch_loss_sum = 0.0
        total_sample = 0
        for (cur_series_tensor, cur_covariate_tensor, cur_real_vals_tensor, cur_demand_tensor, cur_supply_tensor, cur_inventory_tensor, GNN_dataset_1) in data_iter:
        #for data in data_iter:
            print("cur_series_tensor.size:",cur_series_tensor.size())
            print("GNN_dataset.size:",len(GNN_dataset))
            print(GNN_dataset[5])
            batch_size = cur_series_tensor.shape[0]
            seq_len = cur_series_tensor.shape[1]
            horizon_size = cur_covariate_tensor.shape[-1]
            total_sample += batch_size * seq_len * horizon_size
            encoder_optimizer.zero_grad()
            gnn_lstm_optimizer.zero_grad()
            demand_forecast_optimizer.zero_grad()
            gdecoder_optimizer.zero_grad()
            ldecoder_optimizer.zero_grad()
            cGNN_optimizer.zero_grad()
            loss = calc_loss(cur_series_tensor, cur_covariate_tensor, cur_real_vals_tensor, cur_demand_tensor, cur_supply_tensor, cur_inventory_tensor, 
                         encoder, gnn_lstm, demand_forecast, gdecoder, ldecoder,cGNN, batch_size, target_width, A_tensor, B_tensor, GNN_dataset, price, device)
            print("Loss is:",loss)
            
            l1_loss = 0.0
            for param in encoder.parameters():  # 遍历 encoder 的参数
                l1_loss += torch.norm(param, p=1)  # 计算 L1 范数
            for param in gdecoder.parameters():  # 遍历 gdecoder 的参数
                l1_loss += torch.norm(param, p=1)  # 计算 L1 范数
            for param in ldecoder.parameters():  # 遍历 ldecoder 的参数
                l1_loss += torch.norm(param, p=1)  # 计算 L1 范数
            # 将 L1 正则化项添加到损失函数中
            #print("正则项损失为：",l1_loss)
            loss += lambda_l1 * l1_loss
                
            loss.backward()
            encoder_optimizer.step()
            gnn_lstm_optimizer.step()
            demand_forecast_optimizer.step()
            gdecoder_optimizer.step()
            ldecoder_optimizer.step()
            cGNN_optimizer.step()
            epoch_loss_sum += loss.item()
        epoch_loss_mean = epoch_loss_sum / total_sample
        scheduler1.step()
        scheduler1_1.step()
        scheduler1_2.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()
        # if (i + 1) % 5 == 0:
        #     print(f"epoch_num {i + 1}, current loss is: {epoch_loss_mean}")