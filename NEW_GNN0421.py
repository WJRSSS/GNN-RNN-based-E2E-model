import gurobipy as gp
from gurobipy import GRB
from gurobipy import read
import numpy as np
import re
from collections import defaultdict
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os

# ====================== 参数设置 ======================
price_data = pd.read_excel('./32SKU/price_32.xlsx')
target_width = price_data.shape[0]
SKUS = []
for i in range(target_width):
    SKUS.append(price_data.iloc[i][0])
prices = {}
for i in range(target_width):
    prices[SKUS[i]] = price_data.iloc[i][1]
# SKUS = ['EAEZAIN9501', 'EAEZAIN9501EZ3', 'NAEZAIN9501',
#         'NAEZAIN9501EZ', 'NAEZAIN9501EZ2', 'NAEZAIN9501EZ3']

sku_index = {sku: i for i, sku in enumerate(SKUS)}
TOTAL_PERIODS = 360
window_size = 7
num_windows = TOTAL_PERIODS - window_size + 1  # 滑动窗口的数量
TRAIN_RATIO = 0.8  # 训练集80%
np.random.seed(42)
torch.manual_seed(42)

# 加载Excel数据
# df_demand = pd.read_excel('demand_0416_6SKU.xlsx', index_col=None)
# df_supply = pd.read_excel('supply_0416_6SKU.xlsx', index_col=None)
# df_inventory = pd.read_excel('inventory_0416_6SKU.xlsx', index_col=None)

#32个sku
df_demand = pd.read_excel('./32SKU/demand_top32.xlsx')
df_supply = pd.read_excel('./32SKU/supply_top32.xlsx')
df_inventory = pd.read_excel('./32SKU/inventory_top32.xlsx')

# 供给量
supply = df_supply.set_index('ID').to_dict(orient='index')
# 销售量
demand = df_demand.set_index('ID').to_dict(orient='index')
# 库存日报
inventory = df_inventory.set_index('ID').to_dict(orient='index')

# 成本参数设置
# prices = {
#     'EAEZAIN9501': 65, 'EAEZAIN9501EZ3': 70, 'NAEZAIN9501': 69,
#     'NAEZAIN9501EZ': 84, 'NAEZAIN9501EZ2': 89, 'NAEZAIN9501EZ3': 74,
#     'NAEZAIN9501EZ4': 79, 'NAEZAIN9502': 74
# }

# 计算各项成本
trans_cost = {}
for s1 in SKUS:
    for s2 in SKUS:
        if s1 != s2:
            # 改造成本 = 价格差的20%/6
            trans_cost[(s1, s2)] = 1.5

holding_cost_factor = 0.068
shortage_cost_factor = 0.3
M = 1e5
# 库存持有成本和缺货成本
holding_cost = {s: prices[s] * holding_cost_factor for s in SKUS}
shortage_cost = {s: prices[s] * shortage_cost_factor for s in SKUS}
# 输出目录准备
mps_dir = "mps_files_6SKU"
os.makedirs(mps_dir, exist_ok=True)

# ====================== 1. MILP建模与求解 ======================
for w in range(num_windows):
    print(f"正在处理周期 {w+1}/{num_windows}...")

    model = gp.Model(f"SKU_Transformation_Week_{w}")
    start_day = w
    end_day = w + window_size

    # 初始库存为前一天的库存
    if start_day == 0:
        current_inv = {s: df_inventory.loc[0, s] for s in SKUS}
    else:
        current_inv = {s: df_inventory.loc[start_day - 1, s] for s in SKUS}

    # ----------------------------
    # 定义变量
    # ----------------------------
    weekly_trans_vars = model.addVars(
        [(s1, s2) for s1 in SKUS for s2 in SKUS if s1 != s2 ],
        name="Weekly_Trans", vtype=GRB.INTEGER)
    for (s1, s2), var in weekly_trans_vars.items():
        var.VarName = f"Weekly_Trans_{s1}_{s2}_w{w}"
    inv_vars = model.addVars(SKUS, name="Inv", vtype=GRB.INTEGER ,lb=0)
    short_vars = model.addVars(SKUS, name="Short", vtype=GRB.INTEGER ,lb=0)
    # delta_vars = model.addVars(SKUS, name="IsShort", vtype=GRB.BINARY)
    # ----------------------------
    # 约束1：库存平衡
    # ----------------------------
    weekly_supply = {s: 0 for s in SKUS}
    weekly_demand = {s: 0 for s in SKUS}
    for d in range(window_size):
        day = start_day + d
        idx = w * window_size + d
        for s in SKUS:
            weekly_supply[s] += df_supply.loc[day, s]
            weekly_demand[s] += df_demand.loc[day, s]

    for s in SKUS:
        trans_in = gp.quicksum(weekly_trans_vars[s2, s] for s2 in SKUS if s2 != s)
        trans_out = gp.quicksum(weekly_trans_vars[s, s2] for s2 in SKUS if s2 != s)

        # TS_s= (current_inv[s] + weekly_supply[s] + trans_in - trans_out)
        # # 库存下限
        # model.addConstr(inv_vars[s] >= TS_s - weekly_demand[s])
        # model.addConstr(inv_vars[s] <= (TS_s - weekly_demand[s]) + M * (1 - delta_vars[s]))
        # model.addConstr(short_vars[s] >= weekly_demand[s] - TS_s)
        # model.addConstr(short_vars[s] <= (weekly_demand[s] - TS_s) + M * delta_vars[s])

        model.addConstr(
            current_inv[s] + weekly_supply[s] + trans_in + short_vars[s] ==
            weekly_demand[s] + inv_vars[s] + trans_out,
            name=f"WeeklyBalance_{s}_{w}"
        )
    # ----------------------------
    # 约束2：缺货不能为负
    # ----------------------------
    model.addConstrs((short_vars[s] >= 0 for s in SKUS), name="Shortage_NonNegative")
    # ----------------------------
    # 约束3：高配不能转低配
    # ----------------------------
    for s1 in SKUS:
        for s2 in SKUS:
            if sku_index[s1] > sku_index[s2]:
                model.addConstr(weekly_trans_vars[s1, s2] == 0, name=f"No_Downgrade_{s1}_{s2}_{w}")
    # ----------------------------
    # 目标函数
    # ----------------------------
    revenue = gp.quicksum(
        prices[s] * (weekly_demand[s] - short_vars[s]) for s in SKUS
    )
    cost = (
        gp.quicksum(weekly_trans_vars[s1, s2] * trans_cost[s1, s2]
                    for s1 in SKUS for s2 in SKUS if s1 != s2) +
        gp.quicksum(inv_vars[s] * holding_cost[s] for s in SKUS ) +
        gp.quicksum(short_vars[s] * shortage_cost[s] for s in SKUS )
    )
    model.setObjective(revenue - cost, GRB.MAXIMIZE)

    # ----------------------------
    # 模型优化与导出
    # ----------------------------
    model.setParam('OutputFlag', 0)
    model.write(f"{mps_dir}/MILP_week_{w}.mps")
    model.optimize()
    model.write(f"{mps_dir}/MILP_week_{w}.sol")  # 保存解文件

    print(f"MILP solved successfully. Obj: {model.ObjVal:.2f}")
    # 获取所有变量值
    var_dict = {v.VarName: v.X for v in model.getVars()}

torch.manual_seed(42)
np.random.seed(42)

MPS_DIR = f"{mps_dir}/MILP_week_{w}.mps"
SOL_DIR = f"{mps_dir}/MILP_week_{w}.sol"

# 2. 解析 MPS 文件函数
def parse_mps_file(file_path):
    model = read(MPS_DIR)
    variables = [var.VarName for var in model.getVars()]
    constraints = [con.ConstrName for con in model.getConstrs()]
    edges = []
    for i, var in enumerate(variables):
        for j, con in enumerate(constraints):
            coef = np.random.choice([0, 1], p=[0.7, 0.3]) * np.random.uniform(0.1, 1.0)
            if coef != 0:
                edges.append((i, j, coef))
    return variables, constraints, edges

# 3. 解析 SOL 文件函数
def parse_sol_file(file_path,week_idx):
    variable_values = {}
    obj_value = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 获取目标函数值
            if line.startswith("# Objective value"):
                obj_value = float(line.split('=')[1].strip())

            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 2:
                continue
            var_name, value_str = parts
            try:
                value = float(value_str)
                variable_values[var_name] = value
            except ValueError:
                continue

    return variable_values, obj_value

# 4. 构建 PyG 图结构
def safe_normalize_graph_index(x, edge_index):
    nodes_used = torch.unique(edge_index)
    id_map = {old_id.item(): new_id for new_id, old_id in enumerate(nodes_used)}
    new_edge_index = torch.stack([
        torch.tensor([id_map[i.item()] for i in edge_index[0]]),
        torch.tensor([id_map[i.item()] for i in edge_index[1]])
    ], dim=0)
    new_x = x[nodes_used]

    return new_x, new_edge_index

def build_graph_from_mps_sol(mps_path, sol_path,period_id):
    variables, constraints, edges = parse_mps_file(mps_path)
    varname_to_idx = {var: idx for idx, var in enumerate(variables)}
    values, obj = parse_sol_file(sol_path, period_id)
    # print(f"[DEBUG] Week {period_id} parsed {len(values)} variables, example: {list(values.items())[:5]}")

    num_vars = len(variables)
    num_cons = len(constraints)
    num_nodes = num_vars + num_cons

    edge_index = [[], []]
    edge_attr = []

    for var_idx, con_idx, coef in edges:
        edge_index[0].append(var_idx)
        edge_index[1].append(num_vars + con_idx)
        edge_attr.append([coef])
        edge_index[0].append(num_vars + con_idx)
        edge_index[1].append(var_idx)
        edge_attr.append([coef])

    x = torch.eye(num_nodes, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y_values = [0.0] * num_vars

    # 获取SKU——pairs
    for var_name, value in values.items():
        if var_name in varname_to_idx:
            idx = varname_to_idx[var_name]
            y_values[idx] = value
    y_tensor = torch.tensor(y_values, dtype=torch.float)

    var_names = variables
    var_names += [''] * num_cons

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_tensor,
    )
    data.period_id = torch.tensor([period_id], dtype=torch.long)
    data.milp_obj = obj
    data.var_names = var_names
    return data

# for test （可删）
# for w in range(num_windows):
#     MPS_DIR = f"{mps_dir}/MILP_week_{w}.mps"
#     SOL_DIR = f"{mps_dir}/MILP_week_{w}.sol"
#     print(build_graph_from_mps_sol(MPS_DIR, SOL_DIR,w))

# 5. 构建数据集
data_list = []
for i in range(178):
    MPS_DIR = f"{mps_dir}/MILP_week_{i}.mps"
    SOL_DIR = f"{mps_dir}/MILP_week_{i}.sol"
    graph_data = build_graph_from_mps_sol(MPS_DIR, SOL_DIR,period_id=i)
    data_list.append(graph_data)

# 6. 数据划分与加载
train_data = data_list[:int(0.8 * len(data_list))]
test_data = data_list[int(0.8 * len(data_list)):]
# 输出检查格式（可删）
# print(train_data)
# print(train_data[0].x.shape[1])
# print(len(train_data))
# print(test_data)
# print(len(test_data))

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)


# 7. 定义 GNN 模型
class MILPGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MILPGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.global_pool = global_mean_pool
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        # print(">>> Entering forward")
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(f"x.shape[0] = {x.size(0)}")  # 打印节点数
        # print(f"edge_index.shape() = {edge_index.shape}")  # 打印最大的节点编号
        # print(f"edge_index.max() = {edge_index.max().item()}")  # 打印最大的节点编号
        # print("edge_index[0].min() =", edge_index[0].min().item())
        # print("edge_index[1].min() =", edge_index[1].min().item())
        # print("edge_index[0].max() =", edge_index[0].max().item())
        # print("edge_index[1].max() =", edge_index[1].max().item())
        # print("edge_index.max() =", edge_index.max().item())  # 所有节点的最大编号
        # print("edge_index.unique().numel() =", edge_index.unique().numel())  # 有多少个节点被用到了
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.global_pool(x, batch)
        x = self.fc(x)
        return x  # shape: [batch_size, out_channels]

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MILPGNN(in_channels=train_data[0].x.shape[1], hidden_channels=64, out_channels=42).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

# 8. 定义损失函数
def custom_loss(pred, target, milp_obj, big_M=180000):

    alpha=0.5
    mse = F.mse_loss(pred, target)
    penalty = big_M - milp_obj
    penalty_mean = penalty.mean()
    return alpha * mse + (1-alpha) * penalty_mean
# def custom_loss(pred, target, milp_obj):
#     mse = F.mse_loss(pred, target, reduction='mean')  #  返回一个标量
#     return mse

# 9. 训练与测试逻辑
def train_model(model, train_loader, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            batch_size = batch.num_graphs
            y = batch.y.view(-1, 42)

            # print(f"output.shape: {output.shape}")
            # print(f"batch.y.shape: {y.shape}")
            # print(f"expected shape: ({batch_size}, 72)")

            assert output.shape == y.shape
            assert output.shape[0] == batch_size
            assert output.shape[1] == 42
            loss = custom_loss(output, y, batch.milp_obj)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}")

# 为测试，可删除
# train_model(model, train_loader, optimizer, epochs=5)

def evaluate_model(model, test_loader, epochs=5):
    for epoch in range(epochs):
        model.eval()
        test_loss = 0
        for batch in test_loader:
            output = model(batch)  # 应该返回 [batch_size, 56]
            batch_size = batch.num_graphs
            y = batch.y.view(-1, 42)
            assert output.shape == y.shape
            assert output.shape[0] == batch_size
            assert output.shape[1] == 42
            loss = custom_loss(output, y, batch.milp_obj)
            test_loss += loss.item()
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss / len(test_loader):.4f}")
    return test_loss / len(test_loader)


# 开始训练与评估
train_model(model, train_loader, optimizer, epochs=5)
test_loss = evaluate_model(model, test_loader)

# *******
sol_variable_names = []
with open(f"{mps_dir}/MILP_week_0.sol", "r") as f:
    for line in f:
        if line.startswith("Weekly_Trans"):
            var_name = line.split()[0]
            sol_variable_names.append(var_name)
sol_variable_names = sol_variable_names[:30]

def export_results_to_excel(model, data_loader, device, filename):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch)  # shape: [num_variable_nodes_total, 1]
            out = out.view(-1)  # shape: [num_nodes]
            target = batch.y.view(-1)  # shape: [num_nodes]

            period_ids = batch.period_id.view(-1).tolist()
            num_graphs = batch.num_graphs
            assert out.shape[0] % num_graphs == 0, "每图变量数不一致"
            vars_per_graph = out.shape[0] // num_graphs

            for i in range(num_graphs):
                start = i * vars_per_graph
                end = (i + 1) * vars_per_graph

                period = int(period_ids[i])
                target_values = target[start:end].tolist()
                pred_values = out[start:end].tolist()

                row = [f"第{period}周"] + target_values + pred_values
                results.append(row)

    # 创建列名
    # 输出56个变量求解值和预测值（三选一）
    columns = ["周期编号"]
    columns += [f"target_{i}" for i in range(30)]
    columns += [f"pred_{i}" for i in range(30)]
    # 输出72个变量求解值和预测值（三选一）
    # columns = ['周期编号'] + [f'target_{i}' for i in range(72)] + [f'pred_{i}' for i in range(72)]

    # 转为 DataFrame 并写入 Excel
    # df = pd.DataFrame(results, columns=columns)
    df = pd.DataFrame(results, columns=["周期编号"] + [f"target_{i}" for i in range(42)] + [f"pred_{i}" for i in range(42)])
    df = df[columns]
    df.to_excel(filename, index=False)
    print(f"✅ 预测结果已保存至 {filename}")

export_results_to_excel(model, test_loader, device, filename="gnn_predictions_0412_6SKU.xlsx")
export_results_to_excel(model, train_loader, device, filename="gnn_train_predictions_6SKU.xlsx")

# 结果附录
# alpha=0.5
# Epoch 1, Train Loss: 60069.5788
# Epoch 2, Train Loss: 60192.8394
# Epoch 3, Train Loss: 60567.1643
# Epoch 4, Train Loss: 60688.4067
# Epoch 5, Train Loss: 60117.8264
# Epoch 1, Test Loss: 61045.8773
# alpha=1
# Epoch 1, Train Loss: 39445.2434
# Epoch 2, Train Loss: 39734.1462
# Epoch 3, Train Loss: 40212.8111
# Epoch 4, Train Loss: 40513.4500
# Epoch 5, Train Loss: 39367.5285
# Epoch 1, Test Loss: 30379.2483
# alpha=0
# Epoch 1, Train Loss: 80693.9154
# Epoch 2, Train Loss: 80651.5334
# Epoch 3, Train Loss: 80921.5169
# Epoch 4, Train Loss: 80863.3655
# Epoch 5, Train Loss: 80868.1283
# Epoch 1, Test Loss: 91712.5094
# alpha=0.9
# Epoch 1, Train Loss: 43570.1102
# Epoch 2, Train Loss: 43825.8840
# Epoch 3, Train Loss: 44283.6809
# Epoch 4, Train Loss: 44548.4402
# Epoch 5, Train Loss: 43517.5870
# Epoch 1, Test Loss: 36512.5730
