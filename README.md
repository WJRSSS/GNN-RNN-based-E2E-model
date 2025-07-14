# GNN-RNN-based-E2E-model
## 以下文件为当前版本模型核心文件
### 真实数据实验.ipynb：
主执行文件，在jupyter中运行
### MQRNN.py data.py
MQRNN结构初始化以及模型训练完毕后的预测部分
### train.py
核心训练文件，包括训练以及损失函数的定义
### GNN_LSTM.py Decoder.py TwoLayerFC.py
分别定义了模型Teacher部分的Encoder、Decoder结构以及Student部分
### Student Model.py
定义了学生模型结构、训练及预测过程
### Demand Forecast.py
定义了需求预测模块的结构
### projection.py
定义projectnet结构

## 以下文件在过往版本的代码中提供了支持，当前版本未使用
### Encoder.py
旧版本（未整合GNN）Encoder结构
### GNN_func.py 
旧版本GNN结构
### NEW_GNN0329_data.py NEW_GNN0421_data.py
GNN训练代码，新版本中已整合到主执行文件中，不再调用这两个文件
### INV_PJNET_1217.py
projectNet源代码，现已整合至projection.py中
### GNN.ipynb 输出改配标签.ipynb
个人测试所用执行文件，无实际用途

