import gurobipy as gp
import pandas as pd

# 读取Excel文件
df1 = pd.read_excel('demand_1217.xlsx')
df2 = pd.read_excel('supply_1217.xlsx')
df3 = pd.read_excel('inventory_1217.xlsx')

# 周期数
periods = 30
window_size = 7
num_windows = periods - window_size + 1  # 滑动窗口的数量

# 产品种类
products = ['A', 'B']

# 供给量
supply = df2.set_index('ID').to_dict(orient='index')
# 销售量
demand = df1.set_index('ID').to_dict(orient='index')
# 库存日报
inventory_s = df3.set_index('ID').to_dict(orient='index')

# 销售价格
prices = {'A': 110, 'B': 100,}
# 改造成本系数（价格差的20%/6）
transformation_cost_factor = 0.2*1/6
# 库存持有成本和缺货成本
holding_cost = 10
shortage_cost = 15

# 初始化结果 DataFrame
results = pd.DataFrame(columns=['Time', 'A_to_B', 'B_to_A'])

for tm in range(1, num_windows + 1):
    # 初始化模型
    mdl = gp.Model(name='Production_Order_Model')

    # 改造量
    transformation = {(p1, p2, t): mdl.addVar(vtype=gp.GRB.INTEGER, name=f'Transformation_{p1}_{p2}_{t}', lb=0)
                      for p1 in products for p2 in products if p1 != p2 for t in range(periods + 1)}

    # 库存量（每周期末的库存）
    inventory_e = {(t, p): mdl.addVar(vtype=gp.GRB.INTEGER, name=f'FinalInventory_{t}_{p}', lb=0)
                       for t in range(periods + 1) for p in products}

    # 缺货量
    shortages = {(t, p): mdl.addVar(vtype=gp.GRB.INTEGER, name=f'Shortage_{t}_{p}', lb=0)
                 for t in range(periods + 1) for p in products}

    # 销售量
    sales = {(t, p): mdl.addVar(vtype=gp.GRB.INTEGER, name=f'Sales_{t}_{p}', lb=0)
             for t in range(periods + 1) for p in products}

    # 每7天的改装量
    weekly_transformation = {(p1, p2, t): mdl.addVar(vtype=gp.GRB.INTEGER, name=f'WeeklyTrans_{p1}_{p2}_{t}', lb=0)
                             for p1 in products for p2 in products if p1 != p2 for t in range(periods + 1)}


    # 添加约束
    for tn in range(1, num_windows + 1):  # 设置为周期7天
        for p in products:
            #销量不能大于需求：对每个产品而言，其未来一周的销量不能大于未来一周的需求
            mdl.addConstr(gp.quicksum(sales[(t, p)] for t in range(tn, tn + window_size)) <= sum(demand[t][p] for t in range(tn, tn + window_size)), name=f'Sales_Limit_by_Demand_{tn}_{p}')
            #销量不能大于供给：对每个产品而言，其未来一周的销量不能大于未来一周的供给（供给包括外部供给以及净改配流入量）
            mdl.addConstr(gp.quicksum(sales[(t, p)] for t in range(tn, tn + window_size)) <=
                           sum(supply[t][p] for t in range(tn, tn + window_size)) + inventory_s[tn][p] +
                           gp.quicksum(transformation[(q, p, t)] for q in products if q != p for t in range(tn, tn + window_size)) -
                           gp.quicksum(transformation[(p, q, t)] for q in products if q != p for t in range(tn, tn + window_size)) ,
                           name=f'Sales_Limit_by_Supply_Plus_Inventory_Plus_Transformation_{tn}_{p}')

    # 每个周期末的库存水平：定义每个周期的期末库存水平 = 期初库存水平 - 销量 + 供给（供给包括外部供给以及净改配流入量）
    for tn in range(1, num_windows + 1):
        for p in products:
            mdl.addConstr(inventory_e[tn, p] == inventory_s[tn][p] +
                           sum(supply[t][p] for t in range(tn, tn + window_size)) -
                           gp.quicksum(sales[t, p] for t in range(tn, tn + window_size)) +
                           gp.quicksum(transformation[(q, p, t)] for q in products if q != p for t in range(tn, tn + window_size)) -
                           gp.quicksum(transformation[(p, q, t)] for q in products if q != p for t in range(tn, tn + window_size)) ,
                           name=f'Inventory_Level_{tn}_{p}')

    # 每个周期末的缺货量
    for tn in range(1, num_windows + 1):
        for p in products:
            mdl.addConstr(shortages[tn, p] ==
                           sum(demand[t][p] for t in range(tn, tn + window_size)) -
                           gp.quicksum(sales[t, p] for t in range(tn, tn + window_size)),
                           name=f'Shortages_{tn}_{p}')



    # 禁止高配改装低配   （ 只有两个SKU，先不考虑  ）
    # for tn in range(1, num_windows + 1):
    #     for p in products:
    #         for q in products:
    #             if p != q and products.index(p) >= products.index(q):
    #                 mdl.addConstr(gp.quicksum(transformation[(p, q, t)] for t in range(tn, tn + window_size)) == 0, name=f'No_High_To_Low_Transformation_{tn}_{p}')

    # 每个周期（7天）改装量
    for tn in range(1, num_windows + 1):
        for p1 in products:
            for p2 in products:
                if p1 != p2:
                    mdl.addConstr(weekly_transformation[(p1, p2, tn)] == gp.quicksum(transformation[(p1, p2, t)] for t in range(tn, tn + window_size)), name=f'WeeklyTransformation_E_{p1}_{p2}_{tn}')


    # 目标函数（最大化利润）
    profit_expr = mdl.addVar(name="Profit")
    mdl.setObjective(
        #收入：价格 * 销量
        gp.quicksum(
            gp.quicksum(prices[p] * sales[(t, p)] for p in products)
            for t in range(tm, tm + window_size)
        ) -
        #成本一：改配成本
        gp.quicksum(
            transformation_cost_factor * abs(prices[p2] - prices[p1]) *
            weekly_transformation[(p1, p2, tm)]
            for p1 in products for p2 in products if p1 != p2
        ) -
        #成本二：库存成本&缺货成本
        gp.quicksum(
            holding_cost * inventory_e[tm + window_size - 1, p] +
            shortage_cost * shortages[tm + window_size - 1, p]
            for p in products
        ) ,
        sense=gp.GRB.MAXIMIZE
    )

    # 求解
    mdl.optimize()


    # 结果处理
    if mdl.status == gp.GRB.OPTIMAL:
        print(f"第{tm}个窗口解找到。")
        row_data = [f'第{tm}天-第{tm+window_size-1}天', 0, 0]
        for p1 in products:
            for p2 in products:
                if p1 != p2:
                    row_data[1 if p1 == 'A' and p2 == 'B' else
                             2 if p1 == 'B' and p2 == 'A' else 0] = transformation[(p1, p2, tm)].X
        results = results._append({'Time': row_data[0], 'A_to_B': row_data[1], 'B_to_A': row_data[2]}, ignore_index=True)
        #print(results)
        print("max_profit: ", mdl.objVal)
    else:
        print("No solution found.")

# 保存结果到Excel文件
results.to_excel('new_output_1217.xlsx', index=False, engine='openpyxl')
print("数据已成功写入Excel文件。")