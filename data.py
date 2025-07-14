import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def read_df(config: dict):
    """
    This function is for reading the sample testing dataframe
    """
    data_df = pd.read_excel(r'C:\Users\聂伟业\Desktop\sku6_data.xlsx', index_col='report_date')

    time_range = pd.date_range('2020-02-08', '2021-05-08', freq='D')
    series_dict = {}
    i = 0
    for col in data_df.columns:
        series_dict[i] = data_df[col]
        i = i + 1

    df = pd.DataFrame(index=time_range, data=series_dict)


    # 转成序列
    purchase_seq = df[0]
    # 切分出训练集和测试集
    # horizon为30
    real_mean = np.mean(purchase_seq)
    real_std = np.std(purchase_seq)
    scaler = StandardScaler()
    target_df = pd.DataFrame(scaler.fit_transform(df), index=time_range, columns=data_df.columns)

    horizon_size = 30
    covariate_df = pd.DataFrame(index=time_range,
                                data={'dayofweek': time_range.dayofweek,
                                      'month': time_range.month
                                      })
    # 日历数据（协变量）归一化
    for col in covariate_df.columns:
        covariate_df[col] = (covariate_df[col] - np.mean(covariate_df[col])) / np.std(covariate_df[col])

    train_target_df = target_df.iloc[:-horizon_size, :]
    test_target_df = target_df.iloc[-horizon_size:, 0:1]
    train_covariate_df = covariate_df.iloc[:-horizon_size, :]
    test_covariate_df = covariate_df.iloc[-horizon_size:, :]


    return train_target_df, test_target_df, train_covariate_df, test_covariate_df, (real_mean, real_std)
def read_df2(config: dict):
    """
    This function is for reading the sample testing dataframe
    """
    number = config['name']
    data_df = pd.read_excel(r'E:\易点云销量预测\0603数据分析\0607报告\全国20特征数据集.xlsx', index_col='report_date')
    #filtered_df = data_df[data_df['sid'] == number]

    time_range = pd.DatetimeIndex(data_df.index)
    series_dict = {}
    series_dict[0] = data_df['value']

    df = pd.DataFrame(index=time_range, data=series_dict)

    # 转成序列
    purchase_seq = df[0]
    # 切分出训练集和测试集
    # horizon为30
    real_mean = np.mean(purchase_seq)
    real_std = np.std(purchase_seq)
    scaler = StandardScaler()
    target_df = pd.DataFrame(scaler.fit_transform(df), index=time_range, columns=['value'])
    #target_df = df
    horizon_size = config['horizon_size']
    series_dict2 = {}
    series_dict2[0] = time_range.dayofweek
    i = 1
    for col in data_df.columns:
        series_dict2[i] = data_df[col]
        i = i + 1
    series_dict2[1] = time_range.month

    covariate_df = pd.DataFrame(index=time_range,
                                data=series_dict2)
    covariate_df = pd.DataFrame(scaler.fit_transform(covariate_df), index=time_range)
    # 日历数据（协变量）归一化
    # for col in covariate_df.columns:
    #     covariate_df[col] = (covariate_df[col] - np.mean(covariate_df[col])) / np.std(covariate_df[col])

    n = int(30 / horizon_size)
    train_target_df = target_df.iloc[horizon_size:-horizon_size * n - 30]
    test_target_df = target_df.iloc[-horizon_size * n:]
    train_covariate_df = covariate_df.iloc[:-(n + 1) * horizon_size - 30, :]
    test_covariate_df = covariate_df.iloc[-(n + 1) * horizon_size:-horizon_size, :]
    return train_target_df, test_target_df, train_covariate_df, test_covariate_df, (real_mean, real_std), n

def read_df3(config: dict):
    """
    This function is for reading the sample testing dataframe
    """
    data_df = pd.read_excel(r'C:\Users\聂伟业\Desktop\无人机数据.xlsx', index_col='report_date')

    time_range = (data_df.index)
    series_dict = {}
    series_dict[0] = data_df['value']

    df = pd.DataFrame(index=time_range, data=series_dict)


    # 转成序列
    purchase_seq = df[0]
    # 切分出训练集和测试集
    # horizon为30
    real_mean = np.mean(purchase_seq)
    real_std = np.std(purchase_seq)
    scaler = StandardScaler()
    target_df = pd.DataFrame(scaler.fit_transform(df), index=time_range, columns=['value'])

    horizon_size = config['horizon_size']
    series_dict2 = {}
    i = 0
    for col in data_df.columns:
        series_dict2[i] = data_df[col]
        i = i + 1

    covariate_df = pd.DataFrame(index=time_range,
                                data=series_dict2)
    # 日历数据（协变量）归一化
    for col in covariate_df.columns:
        covariate_df[col] = (covariate_df[col] - np.mean(covariate_df[col])) / np.std(covariate_df[col])

    train_target_df = target_df.iloc[:-horizon_size * 6]
    test_target_df1 = target_df.iloc[-horizon_size * 6: -horizon_size * 5]
    test_target_df2 = target_df.iloc[-horizon_size * 5: -horizon_size * 4]
    test_target_df3 = target_df.iloc[-horizon_size * 4: -horizon_size * 3]
    test_target_df4 = target_df.iloc[-horizon_size * 3: -horizon_size * 2]
    test_target_df5 = target_df.iloc[-horizon_size * 2: -horizon_size]
    test_target_df6 = target_df.iloc[-horizon_size:]
    train_covariate_df = covariate_df.iloc[:-horizon_size * 6, :]
    test_covariate_df1 = covariate_df.iloc[-horizon_size * 6: -horizon_size * 5, :]
    test_covariate_df2 = covariate_df.iloc[-horizon_size * 5: -horizon_size * 4, :]
    test_covariate_df3 = covariate_df.iloc[-horizon_size * 4: -horizon_size * 3, :]
    test_covariate_df4 = covariate_df.iloc[-horizon_size * 3 : -horizon_size * 2, :]
    test_covariate_df5 = covariate_df.iloc[-horizon_size * 2 : -horizon_size, :]
    test_covariate_df6 = covariate_df.iloc[-horizon_size:, :]
    return train_target_df, test_target_df1, test_target_df2, test_target_df3, test_target_df4, test_target_df5, test_target_df6, train_covariate_df, test_covariate_df1, test_covariate_df2, test_covariate_df3 ,test_covariate_df4 ,test_covariate_df5 ,test_covariate_df6 , (real_mean, real_std)





def loss_result(target_width, predict_dict, supply_df, inventory_df, demand_df, price):
    seq_len = len(predict_dict)
    Net_income = [[0 for _ in range(target_width)] for _ in range(seq_len)]
    Cost_sum = [[0 for _ in range(target_width)] for _ in range(seq_len)]
    predict_dict = np.array(predict_dict)
    supply_df = supply_df[:seq_len]
    inventory_df = inventory_df[:seq_len]
    demand_df = demand_df[:seq_len]

    
    transformation_cost = 0.2
    Net_income = np.array(Net_income)
    Cost_sum = np.array(Cost_sum)
    #净改配流入
    for i in range(target_width):
        inc = 0
        out = 0
        for j in range(target_width):
            if j < i:
                inc += predict_dict[:,j * (target_width - 1) + i - 1] 
            if j == i:
                continue
            if j > i:
                inc += predict_dict[:, j * (target_width - 1) + i] 
        #inc = inc - Ldecoder_output_PJ[:,:,:,:,i * (target_width - 1) + i]
        for k in range(target_width - 1):
            if k < i:  
                out += predict_dict[:, i * (target_width - 1) + k] * abs(price[i]-price[k]) * transformation_cost
            if k >= i:
                out += predict_dict[:, i * (target_width - 1) + k] * abs(price[i]-price[k + 1]) * transformation_cost
        
        Net_income[:,i] = inc - out
        Cost_sum[:,i] = out


    #计算供给：总供给=期初库存+当期供给+净改配流入
    supply = supply_df + Net_income + inventory_df
    supply = np.clip(supply,0,float('inf'))
    #取supply和demand的最小值
    sales = np.minimum(supply,demand_df)
    sales = np.array(sales)

    for j in range(target_width):
        sales[:,j] = sales[:,j] * price[j]
    #print(cost)
    # print("收入",sales)
    # print("支出",Cost_sum)
    profit = sales - Cost_sum
    errors = 1 / (1 + np.exp(profit/1000))
    return np.sum(errors)/seq_len









class MQRNN_dataset(Dataset):

    def __init__(self,
                 series_df: pd.DataFrame,
                 covariate_df: pd.DataFrame,
                 demand_df:pd.DataFrame,
                 supply_df:pd.DataFrame,
                 inventory_df:pd.DataFrame,
                 GNN_dataset: list,
                 horizon_size: int,
                 quantile_size: int):

        self.series_df = series_df
        self.demand_df = demand_df
        self.supply_df = supply_df
        self.inventory_df = inventory_df
        self.covariate_df = covariate_df
        self.horizon_size = horizon_size
        self.quantile_size = quantile_size
        self.GNN_dataset = GNN_dataset
        full_covariate = []
        covariate_size = self.covariate_df.shape[1]
        self.length = self.covariate_df.shape[0] - self.horizon_size
        #print(f"self.covariate_df.shape[0] : {self.horizon_size}")
        for i in range(1, self.covariate_df.shape[0] - horizon_size + 1):
            cur_covariate = []
            # for j in range(horizon_size):
            cur_covariate.append(self.covariate_df.iloc[i:i + horizon_size, :].to_numpy())
            full_covariate.append(cur_covariate)
        full_covariate = np.array(full_covariate)
        #print(f"full_covariate shape: {full_covariate.shape}")
        full_covariate = full_covariate.reshape(-1, horizon_size * covariate_size)
        self.next_covariate = full_covariate

    def __len__(self):
        try:
            return 1
        except:
            return 1

    def __getitem__(self, idx):

        cur_series = np.array(self.series_df.iloc[: -self.horizon_size, :])
        cur_covariate = np.array(
            self.covariate_df.iloc[:-self.horizon_size, :])  # covariate used in generating hidden states

        covariate_size = self.covariate_df.shape[1]
        # next_covariate = np.array(self.covariate_df.iloc[1:-self.horizon_size+1,:]) # covariate used in the MLP decoders

        real_vals_list = []
        for i in range(1, self.horizon_size + 1):
            real_vals_list.append(
                np.array(self.series_df.iloc[i: self.series_df.shape[0] - self.horizon_size + i, :]))
            
        demand_list = []
        for i in range(1, self.horizon_size + 1):
            demand_list.append(
                np.array(self.demand_df.iloc[i: self.demand_df.shape[0] - self.horizon_size + i, :]))
            
        supply_list = []
        for i in range(1, self.horizon_size + 1):
            supply_list.append(
                np.array(self.supply_df.iloc[i: self.supply_df.shape[0] - self.horizon_size + i, :]))
            
        inventory_list = []
        for i in range(1, self.horizon_size + 1):
            inventory_list.append(
                np.array(self.inventory_df.iloc[i: self.inventory_df.shape[0] - self.horizon_size + i, :]))




        real_vals_array = np.array(real_vals_list)  # [horizon_size, seq_len]
        demand_array = np.array(demand_list)  # [horizon_size, seq_len]
        supply_array = np.array(supply_list)  # [horizon_size, seq_len]
        inventory_array = np.array(inventory_list)  # [horizon_size, seq_len]
        real_vals_array = real_vals_array.T
        demand_array = demand_array.T  # [seq_len, horizon_size]
        supply_array = supply_array.T
        inventory_array = inventory_array.T
        cur_series_tensor = torch.tensor(cur_series)

        #cur_series_tensor = torch.unsqueeze(cur_series_tensor, dim=1)  # [seq_len, 1]
        cur_covariate_tensor = torch.tensor(cur_covariate)  # [seq_len, covariate_size]

        cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor], dim=1)
        next_covariate_tensor = torch.tensor(self.next_covariate)  # [seq_len, horizon_size * covariate_size]

        cur_real_vals_tensor = torch.tensor(real_vals_array)
        cur_real_vals_tensor = cur_real_vals_tensor.permute(1,2,0)

        cur_demand_tensor = torch.tensor(demand_array)
        cur_demand_tensor = cur_demand_tensor.permute(1,2,0)

        cur_supply_tensor = torch.tensor(supply_array)
        cur_supply_tensor = cur_supply_tensor.permute(1,2,0)

        cur_inventory_tensor = torch.tensor(inventory_array)
        cur_inventory_tensor = cur_inventory_tensor.permute(1,2,0)

        GNN_dataset = self.GNN_dataset[:]

        return cur_series_covariate_tensor, next_covariate_tensor, cur_real_vals_tensor, cur_demand_tensor, cur_supply_tensor, cur_inventory_tensor, GNN_dataset