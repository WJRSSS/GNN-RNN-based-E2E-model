import gurobipy as gp
from gurobipy import GRB
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
import pandas as pd
from sklearn.model_selection import train_test_split

# ====================== 参数设置 ======================
# price_data = pd.read_excel('./32SKU/price_32.xlsx')
# target_width = price_data.shape[0]
SKUS = ['EAEZAIN9501', 'EAEZAIN9501EZ3', 'NAEZAIN9501',
        'NAEZAIN9501EZ', 'NAEZAIN9501EZ2', 'NAEZAIN9501EZ3',
        'NAEZAIN9501EZ4', 'NAEZAIN9502']
# SKUS = []
# for i in range(target_width):
#     SKUS.append(price_data.iloc[i][0])
# prices = {}
# for i in range(target_width):
#     prices[SKUS[i]] = price_data.iloc[i][1]


TOTAL_PERIODS = 184
TRAIN_RATIO = 0.8  # 训练集80%
np.random.seed(42)
torch.manual_seed(42)

# 加载Excel数据
df_demand = pd.read_excel('demand_0206.xlsx')
df_supply = pd.read_excel('supply_0206.xlsx')
df_inventory = pd.read_excel('inventory_0206.xlsx')
#32个sku
# df_demand = pd.read_excel('./32SKU/demand_top32.xlsx')
# df_supply = pd.read_excel('./32SKU/supply_top32.xlsx')
# df_inventory = pd.read_excel('./32SKU/inventory_top32.xlsx')

# 供给量
supply = df_supply.set_index('ID').to_dict(orient='index')
# 销售量
demand = df_demand.set_index('ID').to_dict(orient='index')
# 库存日报
#initial_inv = df_inventory.set_index('ID').to_dict(orient='index')

# 成本参数设置
prices = {
    'EAEZAIN9501': 65, 'EAEZAIN9501EZ3': 70, 'NAEZAIN9501': 69,
    'NAEZAIN9501EZ': 84, 'NAEZAIN9501EZ2': 89, 'NAEZAIN9501EZ3': 74,
    'NAEZAIN9501EZ4': 79, 'NAEZAIN9502': 74
}

# 计算各项成本
trans_cost = {}
for s1 in SKUS:
    for s2 in SKUS:
        if s1 != s2:
            # 改造成本 = 价格差的20%/6
            trans_cost[(s1, s2)] = abs(prices[s1] - prices[s2]) * 0.2 / 6

holding_cost_factor = 0.068
shortage_cost_factor = 0.3
# 库存持有成本和缺货成本
holding_cost = {s: prices[s] * holding_cost_factor for s in SKUS}
shortage_cost = {s: prices[s] * shortage_cost_factor for s in SKUS}

# ====================== 1. MILP建模与求解 ======================
def build_and_solve_milp():
    print("Building MILP model...")
    print(SKUS)

    # 生成模拟数据
    #demand = {(s, t): np.random.randint(10, 50) for s in SKUS for t in range(TOTAL_PERIODS)}
    #holding_cost = {s: np.random.uniform(0.1, 0.5) for s in SKUS}
    #trans_cost = {(s1, s2): np.random.uniform(1, 3) for s1 in SKUS for s2 in SKUS if s1 != s2}
    #shortage_cost = {s: np.random.uniform(5, 10) for s in SKUS}
    initial_inv = {s: np.random.randint(50, 100) for s in SKUS}

    # 建立模型
    model = gp.Model("SKU_Transformation")

    # 决策变量
    # 在build_and_solve_milp()函数中修改：
    trans_vars = model.addVars(
        [(s1, s2, t) for s1 in SKUS for s2 in SKUS for t in range(TOTAL_PERIODS) if s1 != s2],
        name="Trans",  # 基础名称
        vtype=GRB.CONTINUOUS
    )
    # 手动设置变量名（关键修正）
    for (s1, s2, t), var in trans_vars.items():
        var.VarName = f"Transformation_{s1}_{s2}_t{t}"

    inv_vars = model.addVars(SKUS, range(TOTAL_PERIODS), name="Inv")
    short_vars = model.addVars(SKUS, range(TOTAL_PERIODS), name="Short")

    # 目标函数
    model.setObjective(
        gp.quicksum(trans_vars[s1, s2, t] * trans_cost[s1, s2]
                    for s1, s2, t in trans_vars) +
        gp.quicksum(inv_vars[s, t] * holding_cost[s]
                    for s in SKUS for t in range(TOTAL_PERIODS)) +
        gp.quicksum(short_vars[s, t] * shortage_cost[s]
                    for s in SKUS for t in range(TOTAL_PERIODS)),
        GRB.MINIMIZE
    )

    # 约束条件
    for t in range(TOTAL_PERIODS):
        for s in SKUS:
            prev_inv = initial_inv[s] if t == 0 else inv_vars[s, t - 1]
            model.addConstr(
                inv_vars[s, t] == prev_inv - demand.get((s, t), 0) + supply.get((s, t), 0) -
                gp.quicksum(trans_vars[s, s2, t] for s2 in SKUS if s != s2) +
                gp.quicksum(trans_vars[s1, s, t] for s1 in SKUS if s1 != s) +
                short_vars[s, t],
                name=f"Balance_{s}_{t}"
            )

    # 求解并保存MPS和SOL文件
    model.optimize()
    if model.status == GRB.OPTIMAL:
        model.write("sku_transformation.mps")
        model.write("sku_transformation.sol")  # 保存解文件
        print(f"MILP solved successfully. Obj: {model.ObjVal:.2f}")
        return {v.VarName: v.X for v in model.getVars()}
    else:
        raise RuntimeError("MILP failed to solve")


# ====================== 2. MPS转二分图 ======================
def parse_mps_and_build_graphs(var_values):
    print("Parsing MPS and building graphs...")

    # 1. 解析MPS文件（增加异常处理）
    try:
        with open("sku_transformation.mps", 'r') as f:
            mps_content = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError("MPS文件未找到，请先运行MILP建模部分生成sku_transformation.mps")

    # 2. 改进正则表达式（处理可能的特殊字符）
    ## sku_pattern构造SKU正则匹配模式，提取变量名称
    sku_pattern = "|".join(re.escape(sku) for sku in SKUS)
    var_pattern = re.compile(
        rf"\sTransformation_({sku_pattern})_({sku_pattern})_t(\d+)\s"
    )
    print(f"当前匹配模式: {var_pattern.pattern}")

    ## time_data存储不同时间步的变量及约束关系
    time_data = defaultdict(lambda: {'vars': [], 'constrs': defaultdict(list)})
    current_section = None

    for line in mps_content:
        if line.startswith('COLUMNS'):
            current_section = 'COLUMNS'
        elif line.startswith('ROWS'):
            current_section = 'ROWS'
        elif line.startswith('RHS'):
            break

        if current_section == 'COLUMNS' and line.startswith(' '):
            match = var_pattern.search(line)
            if match:
                p1, p2, t = match.groups()
                t = int(t)
                parts = line.split()
                for i in range(1, len(parts), 2):
                    if parts[i] == "'MARKER'":
                        continue  # 跳过标记行
                    constr = parts[i]
                    coeff = float(parts[i + 1])
                    var_name = parts[0]
                    time_data[t]['vars'].append(var_name)
                    time_data[t]['constrs'][constr].append((var_name, coeff))

    # 3. 验证数据是否被正确提取
    if not time_data:
        raise ValueError("未从MPS文件中提取到任何有效数据，请检查变量命名格式")
    print(f"成功提取 {len(time_data)} 个时间步的数据")

    # 4. 构建图数据集（增加空值检查）
    all_graphs = []
    for t in sorted(time_data.keys()):
        if not time_data[t]['vars']:
            print(f"警告：时间步 {t} 无变量数据，跳过")
            continue

        B = nx.Graph()
        var_nodes = list(set(time_data[t]['vars']))  # 去重
        constr_nodes = list(time_data[t]['constrs'].keys())

        ## 变量=0，约束=1
        B.add_nodes_from(var_nodes, bipartite=0, type='var')
        B.add_nodes_from(constr_nodes, bipartite=1, type='constr')

        for constr, var_coeffs in time_data[t]['constrs'].items():
            for var, coeff in var_coeffs:
                B.add_edge(var, constr, weight=coeff)

        # 转换为PyG Data（增加维度检查）
        ## 变量节点为 [1, t/TOTAL_PERIODS]，约束节点为 [0, t/TOTAL_PERIODS]
        try:
            x = torch.tensor([
                [1, t / TOTAL_PERIODS] if B.nodes[n]['type'] == 'var' else [0, t / TOTAL_PERIODS]
                for n in B.nodes
            ], dtype=torch.float)

            ## 边索引
            edge_index = torch.tensor([
                [list(B.nodes).index(u), list(B.nodes).index(v)]
                for u, v in B.edges
            ]).t().contiguous()

            ## 边权重
            edge_attr = torch.tensor([
                B.edges[u, v]['weight'] for u, v in B.edges
            ], dtype=torch.float).unsqueeze(1)

            # 确保y只对应变量节点，y为真实变量值
            pred_vars = [v for v in B.nodes if B.nodes[v]['type'] == 'var']
            y_true = torch.tensor([var_values[var] for var in var_nodes], dtype=torch.float).view(-1, 1)

            graph_data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=y_true, var_names=pred_vars, time_step=t,
                var_mask=torch.tensor([B.nodes[n]['type'] == 'var' for n in B.nodes], dtype=torch.bool)
            )
            all_graphs.append(graph_data)
        except Exception as e:
            print(f"时间步 {t} 图构建失败: {str(e)}")
            continue

    if not all_graphs:
        raise RuntimeError("未能构建任何有效的图数据")
    return all_graphs

# ====================== 3. GNN模型 ======================
## 第一层输入维度2——>隐藏层维度64，第二层隐藏层64——>64，前向传播
class ConstraintGNN(nn.Module):
    def __init__(self, 
                 batch_size: int,
                 target_width: int,
                 hidden_dim = 64,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.target_width = target_width
        self.conv1 = geom_nn.GCNConv(2, hidden_dim)
        self.conv2 = geom_nn.GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, batch_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 只对变量节点进行预测（bipartite=0的节点）
        var_mask = data.x[:, 0] == 1  # 第一列为1的是变量节点
        var_nodes = torch.where(var_mask)[0]

        # 仅用变量节点进行前向传播
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        # 只选取变量节点的输出
        out = torch.relu(self.fc(x[var_mask]))
        seq_len = int(out.shape[0] / (self.target_width * (self.target_width - 1)))
        #print("seq_len =:",seq_len)
        out_reshape = out.view(seq_len, self.batch_size, self.target_width * (self.target_width - 1))
        return torch.round(out_reshape)


# ====================== 4. 训练与评估 ======================
## 计算模型在测试集表现，计算均方误差MAE和均方根误差RMSE
def calculate_accuracy(model, dataset):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data in dataset:
            out = model(data)
            all_preds.append(out.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())

    preds = np.vstack(all_preds)  # shape: (n_periods*56, 1)
    targets = np.vstack(all_targets)

    # 计算关键指标
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    return {
        'MAE': mae,
        'RMSE': rmse,
        'predictions': preds.reshape(-1, 56),  # shape: (n_periods, 56)
        'targets': targets.reshape(-1, 56)
    }

def train_and_evaluate(dataset):
    # 确保数据集非空
    if not dataset:
        raise ValueError("输入数据集为空，请检查数据解析过程")

    # 划分训练测试集（确保至少有一个样本）
    test_size = max(1, int((1 - TRAIN_RATIO) * len(dataset)))
    train_data, test_data = dataset[:-test_size], dataset[-test_size:]

    print(f"训练集: {len(train_data)} 个图, 测试集: {len(test_data)} 个图")

    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # 初始化模型
    model = ConstraintGNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 训练循环
    print("Training GNN...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                test_loss += criterion(out, batch.y).item()

        # 控制输出频率，减少日志输出量（184周期*100epoch）
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {total_loss / len(train_loader):.4f} | "
                  f"Test Loss: {test_loss / len(test_loader):.4f}")

    # 计算训练集/测试集精度
    train_results = calculate_accuracy(model, train_data)
    test_results = calculate_accuracy(model, test_data)

    # 打印关键指标
    print("\n=== 训练集精度（154周期）===")
    print(f"MAE: {train_results['MAE']:.4f}")
    print(f"RMSE: {train_results['RMSE']:.4f}")

    print("\n=== 测试集精度（30周期）===")
    print(f"MAE: {test_results['MAE']:.4f}")
    print(f"RMSE: {test_results['RMSE']:.4f}")

    return model


def export_test_results(model, test_dataset, var_values, output_file="test_predictions.xlsx"):
    """
    输出测试集预测结果到Excel
    格式：周期数 + 56个SKU对的真实值和预测值（共113列）
    """
    model.eval()

    # 准备SKU对列表（按字母顺序排序保证一致性）
    sku_pairs = sorted([(s1, s2) for s1 in SKUS for s2 in SKUS if s1 != s2],
                       key=lambda x: (x[0], x[1]))

    # 创建结果DataFrame
    results = []

    with torch.no_grad():
        for data in test_dataset:
            # 获取预测值
            preds = model(data).cpu().numpy().flatten()

            # 获取真实值（从var_values中提取）
            true_values = []
            for s1, s2 in sku_pairs:
                var_name = f"Transformation_{s1}_{s2}_t{data.time_step}"
                true_values.append(var_values.get(var_name, 0.0))

            # 构建当前周期的记录
            record = {"周期数": data.time_step}

            # 添加56个SKU对的数据
            for idx, (s1, s2) in enumerate(sku_pairs):
                record[f"{s1}_to_{s2}_真实值"] = true_values[idx]
                record[f"{s1}_to_{s2}_预测值"] = preds[idx]

            results.append(record)

    # 转换为DataFrame并保存
    df_results = pd.DataFrame(results)

    # 按周期排序
    df_results.sort_values("周期数", inplace=True)

    # 保存到Excel
    df_results.to_excel(output_file, index=False)
    print(f"测试集预测结果已保存到 {output_file}")
    return df_results

def export_train_results(model, train_dataset, var_values, output_file="train_predictions.xlsx"):
    """输出训练集预测结果（154周期）"""
    sku_pairs = sorted([(s1, s2) for s1 in SKUS for s2 in SKUS if s1 != s2])

    results = []
    with torch.no_grad():
        for data in train_dataset:
            record = {"周期数": int(data.time_step)}
            preds = model(data).cpu().numpy().flatten().astype(int)

            for idx, (s1, s2) in enumerate(sku_pairs):
                var_name = f"Transformation_{s1}_{s2}_t{data.time_step}"
                record[f"{s1}_to_{s2}_真实值"] = int(round(var_values.get(var_name, 0)))
                record[f"{s1}_to_{s2}_预测值"] = preds[idx]

            results.append(record)

    pd.DataFrame(results).to_excel(output_file, index=False)
    print(f"训练集结果已保存到 {output_file}")

# ====================== 主执行流程 ======================
if __name__ == "__main__":
    # 1. 构建并求解MILP
    var_values = build_and_solve_milp()
    print("SKUS列表:", SKUS)
    var_values = build_and_solve_milp()
    print("变量值示例:", list(var_values.items())[:3])

    # 2. 解析MPS并构建图数据集
    dataset = parse_mps_and_build_graphs(var_values)
    print(f"Total graphs: {len(dataset)} (Train: {int(TRAIN_RATIO * len(dataset))}, "
          f"Test: {len(dataset) - int(TRAIN_RATIO * len(dataset))})")
    print("单个图的维度检查:")
    print("x shape:", dataset[0].x.shape)
    print("y shape:", dataset[0].y.shape)
    print("变量节点数:", dataset[0].var_mask.sum().item())

    # 3. 训练和评估GNN
    model = train_and_evaluate(dataset)
    #print("生成的变量名示例:", [v.VarName for v in list(model.getVars())[:5]])

    # 新增：输出测试集预测结果
    test_dataset = dataset[-int((1 - TRAIN_RATIO) * len(dataset)):]  # 获取测试集
    test_results = export_test_results(model, test_dataset, var_values)
    # 输出训练集结果
    train_dataset = dataset[:int(TRAIN_RATIO * len(dataset))]  # 前154个周期
    export_train_results(model, train_dataset, var_values)

    # 4. 保存模型
    torch.save(model.state_dict(), "gnn_predictor.pt")
    print("Pipeline completed. Model saved.")