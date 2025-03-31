# 恒星光谱特征预测模型

该项目使用深度学习模型预测恒星光谱的关键特征参数（logg, feh, teff）。

## 项目结构
```
├── configs/           # 配置文件
│   ├── config.yaml    # MLP模型配置
│   └── conv1d_config.yaml # Conv1D模型配置
├── models/            # 模型定义
├── utils/             # 工具函数
├── train/             # 训练相关脚本
├── detect.py          # 预测脚本
├── evaluate.py        # 模型评估脚本
├── demo/              # 演示应用
├── data/              # 数据目录
│   ├── train/         # 训练数据集
│   │   ├── features.csv  # 训练特征
│   │   └── labels.csv    # 训练标签
│   └── test/          # 测试数据集
│       ├── features.csv  # 测试特征
│       └── labels.csv    # 测试标签
├── train.py           # 主训练脚本
├── run_train.py       # 快速启动训练脚本
├── run_evaluate.py    # 快速启动评估脚本
└── requirements.txt   # 项目依赖
```

## 特征说明

本项目主要关注三个关键特征：
- `logg`: 表面重力加速度（单位：dex）- 表征恒星表面的重力加速度，与恒星质量和半径相关
- `feh`: 金属丰度（单位：dex）- 恒星大气中金属元素相对于氢的丰度对数比值，通常以太阳为基准
- `teff`: 有效温度（单位：K）- 恒星表面温度，决定了恒星的光谱类型和颜色

## 数据集说明

### 数据格式

数据集分为训练集和测试集，每个集合由特征文件和标签文件组成：

1. **特征文件（features.csv）**：
   - 第一列必须是`obsid`（观测ID），用于唯一标识每个样本
   - 其余列为恒星光谱数据，通常包含数百至数千个波长点的光谱通量值
   - 特征数量（列数）可能较多，从几百到上千不等

2. **标签文件（labels.csv）**：
   - 必须包含`obsid`列，用于与特征文件进行匹配
   - 包含预测目标列：`logg`、`feh`和`teff`
   - 可能包含其他辅助信息列（如观测时间、信噪比等）

### 数据范围

标签数据的典型范围：
- **logg**: 通常在0.0至5.0 dex之间，恒星巨星的值较小，矮星的值较大
- **feh**: 通常在-4.0至+0.5 dex之间，负值表示金属含量低于太阳，正值表示高于太阳
- **teff**: 通常在3,000至10,000 K之间，覆盖从K型红矮星到A型恒星的温度范围

### 数据预处理

在训练前，数据会自动进行以下预处理：

1. **特征归一化**：使用标准化（StandardScaler）方法将特征数据转换为均值为0、标准差为1的分布
2. **标签归一化**：同样使用标准化方法处理标签数据
3. **数据分割**：训练数据会自动分成训练集和验证集（默认80%/20%）

### 数据要求

- 特征文件和标签文件必须有相同数量的样本，且通过`obsid`列一一对应
- 特征数据应该是预处理过的光谱通量值，最好已去除异常值和噪声
- 标签数据应该是经过专业分析得到的恒星物理参数

### 示例数据

特征文件（features.csv）示例：
```
obsid,flux_1,flux_2,flux_3,...,flux_1684
spec_00001,0.954,0.967,0.982,...,1.023
spec_00002,0.871,0.896,0.913,...,0.957
...
```

标签文件（labels.csv）示例：
```
obsid,logg,feh,teff
spec_00001,4.5,-0.2,5800
spec_00002,2.8,-1.4,4200
...
```

## 环境配置
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据准备

确保`data/train/`和`data/test/`目录下分别有特征文件(`features.csv`)和标签文件(`labels.csv`)。两个文件通过`obsid`列进行关联。

### 2. 训练模型

使用默认配置训练MLP模型:
```bash
python run_train.py --config configs/config.yaml
```

或者训练Conv1D模型:
```bash
python run_train.py --config configs/conv1d_config.yaml
```

指定实验名称:
```bash
python run_train.py --config configs/config.yaml --exp-name my_experiment
```

训练结果将保存在`runs/模型名称_时间戳/`目录下，包括：
- 模型权重: `weights/best.pt`
- 损失曲线图: `plots/loss_curve.png`
- 配置文件和日志

### 3. 评估模型

训练完成后，使用以下命令评估模型性能:
```bash
python run_evaluate.py --model-path runs/模型名称_时间戳/weights/best.pt
```

评估结果将保存在`runs/模型名称_时间戳/evaluation/`目录下，包括：
- 评估指标: `evaluation_results.csv`
- 预测结果: `predictions.csv`
- 每个特征的预测vs实际散点图

### 4. 使用训练好的模型进行预测

对新数据进行预测：
```bash
python detect.py --weights runs/模型名称_时间戳/weights/best.pt --input data/test/ --output predictions
```

可以对单个文件或整个目录进行预测：
```bash
# 预测目录中的数据
python detect.py --weights runs/模型名称_时间戳/weights/best.pt --input data/new_data/ --output predictions

# 预测单个文件
python detect.py --weights runs/模型名称_时间戳/weights/best.pt --input data/new_data/features.csv --output predictions
```

生成预测结果的可视化图表：
```bash
python detect.py --weights runs/模型名称_时间戳/weights/best.pt --input data/test/ --output predictions --save-plots
```

预测结果将保存在指定的`output`目录中：
- 预测结果CSV文件：`predictions_时间戳.csv`
- 可视化图表（如果使用`--save-plots`）：`plots/`目录下

## 模型类型
- **MLP** (多层感知机): 适用于简单数据关系
- **Conv1D** (一维卷积神经网络): 适用于序列数据和信号处理

## 配置文件说明

配置文件(configs/config.yaml)允许自定义模型结构和训练参数：

```yaml
# 模型配置
model:
  name: mlp  # 可选: mlp, conv1d
  input_dim: 64
  hidden_dims: [128, 256, 128]
  output_dim: 3  # 预测三个特征: logg, feh, teff
  dropout_rate: 0.2

# 数据配置
data:
  data_size: 1000
  seq_len: 64
  feature_dim: 1
  val_split: 0.2
  test_split: 0.1
  num_workers: 4
  id_column: obsid  # 用于关联特征和标签文件的ID列
  labels:  # 预测目标列表
    - logg  # 表面重力
    - feh   # 金属丰度
    - teff  # 有效温度
  use_all_features: true  # 使用特征文件中的所有特征

# 训练配置
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  loss: mse  # 可选: mse, mae, bce, ce
  optimizer: adam  # 可选: adam, sgd, rmsprop
  scheduler: cosine  # 可选: step, cosine, plateau, None
```

## 更新日志

- **2023-03-30**: 
  - 添加了配置文件中的特征选项
  - 实现从实际数据集加载数据
  - 添加评估脚本和可视化功能
  - 支持通过obsid关联特征和标签文件
  
- **最新更新**:
  - 修复了detect.py脚本，使其能够从实际数据集加载数据
  - 支持使用obsid列进行预测结果输出
  - 更新了预测可视化功能，支持多特征预测结果展示
  - 改进了项目文档，添加了详细的使用说明
  - 更新了requirements.txt，添加了所有必要的依赖项

## 使用TensorBoard监控训练过程

本框架支持使用TensorBoard监控训练过程，训练日志会自动保存在实验目录下的`logs`文件夹内。

启动TensorBoard：

```bash
tensorboard --logdir=runs/实验名称/logs
```

然后在浏览器中访问 http://localhost:6006 查看训练过程中的：

- 训练损失和验证损失
- 每个特征的MSE损失
- 学习率变化
- 模型架构图
- 最佳模型性能

## 评估模型

使用以下命令评估已训练的模型：

```bash
python evaluate.py --model-path runs/模型名_时间戳/weights/best.pt --config configs/config.yaml
```

评估结果将保存在`results/`目录下。

## 使用模型进行预测

使用以下命令对新数据进行预测：

```bash
python detect.py --weights runs/模型名_时间戳/weights/best.pt --input data/test/features.csv --output predictions
```

可以添加`--save-plots`参数生成预测可视化图表。

## 支持的模型

- MLP：多层感知机模型
- Conv1D：一维卷积神经网络

## 注意事项

- 请确保数据已经适当清洗和预处理
- 模型训练过程会自动对特征和标签进行归一化处理 