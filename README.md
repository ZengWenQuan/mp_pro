# Deep Learning Project

这是一个基于 PyTorch 的深度学习项目，支持多种神经网络模型进行预测任务。

## 功能特点

- 支持多种深度学习模型：

  - MLP (多层感知机)
  - Conv1D (一维卷积神经网络)
  - LSTM (长短期记忆网络)
  - Transformer (变换器模型)
  - Autoencoder (自编码器)
  - MPBDNet (多路径双分支网络)
  - Autoformer (自相关变换器)
  - Informer (信息增强变换器)
- 完整的训练和评估流程：

  - 支持断点恢复训练（使用 --resume_from）
  - 自动保存最佳模型
  - 每10轮保存一次检查点
  - 支持早停机制
  - TensorBoard 可视化支持
- 灵活的配置系统：

  - 使用 YAML 配置文件
  - 可配置模型结构、训练参数等
  - 支持命令行参数覆盖

## 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
├── configs/           # 配置文件
│   ├── mlp_config.yaml    # MLP模型配置
│   ├── lstm_config.yaml   # LSTM模型配置
│   ├── conv1d_config.yaml # Conv1D模型配置
│   ├── transformer_config.yaml # Transformer模型配置
│   ├── mpbdnet.yaml   # MPBDNet模型配置
│   ├── autoformer.yaml # Autoformer模型配置
│   └── informer.yaml  # Informer模型配置
├── models/            # 模型定义
├── utils/             # 工具函数
├── train/             # 训练相关脚本
├── detect.py          # 预测脚本
├── evaluate.py        # 模型评估脚本
├── demo/              # 演示应用
├── data/              # 默认数据目录
│   ├── train/         # 默认训练数据集目录
│   │   ├── features.csv  # 训练特征
│   │   └── labels.csv    # 训练标签
│   └── test/          # 默认测试数据集目录
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

   - 第一列必须是 `obsid`（观测ID），用于唯一标识每个样本
   - 其余列为恒星光谱数据，通常包含数百至数千个波长点的光谱通量值
   - 特征数量（列数）可能较多，从几百到上千不等
2. **标签文件（labels.csv）**：

   - 必须包含 `obsid`列，用于与特征文件进行匹配
   - 包含预测目标列：`logg`、`feh`和 `teff`
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

- 特征文件和标签文件必须有相同数量的样本，且通过 `obsid`列一一对应
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

本项目提供了智能安装脚本 `setup.py`，可以根据您的环境自动选择合适的依赖项。

### 特点

- 自动检测已安装的库，避免重复安装
- 自动检测GPU/CUDA可用性
  - 若检测到GPU，则安装支持CUDA的PyTorch版本
  - 若无GPU，则安装CPU版本的PyTorch
- 只安装缺失的依赖，提高安装效率

### 使用方法

只需运行以下命令：

```bash
python setup.py
```

### 安装过程

1. 脚本会先检测您的环境是否有GPU
2. 检查已安装的库及其版本
3. 只安装缺失或版本不符的库
4. 对于PyTorch相关库，根据GPU可用性选择合适版本
5. 安装完成后验证PyTorch安装情况

### 手动安装

如果您想手动安装，可以使用以下命令：

```bash
# 使用GPU版本（需要NVIDIA GPU和CUDA）
pip install -r requirements.txt

# 使用CPU版本
pip install torch>=1.10.0+cpu torchvision>=0.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --ignore-installed torch torchvision
```

## 依赖说明

主要依赖包括：

- PyTorch >= 1.10.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Scikit-learn >= 0.24.0
- TensorBoard >= 2.6.0
- PyYAML >= 5.4.0
- Progressbar2 >= 3.53.0

## 使用方法

### 1. 数据准备

数据路径可以在配置文件（configs/*.yaml）中指定：

```yaml
# Data Configuration
data:
  train_dir: data/train     # 训练数据目录
  test_dir: data/test      # 测试数据目录
  features_file: features.csv  # 特征文件名
  labels_file: labels.csv     # 标签文件名
```

确保在指定的训练和测试目录下分别有特征文件和标签文件。两个文件通过 `obsid` 列进行关联。

### 2. 训练模型

使用默认配置训练MLP模型:

```bash
python run_train.py --config configs/mlp.yaml
```

或者训练其他模型:

```bash
python run_train.py --config configs/lstm.yaml
python run_train.py --config configs/conv1d.yaml
python run_train.py --config configs/transformer.yaml
python run_train.py --config configs/mpbdnet.yaml
python run_train.py --config configs/autoformer.yaml
python run_train.py --config configs/informer.yaml
```

指定实验名称:

```bash
python run_train.py --config configs/mlp.yaml --exp-name my_experiment
```

训练结果将保存在 `runs/模型名称_时间戳/`目录下，包括：

- 模型权重: `weights/best.pt`
- 损失曲线图: `plots/loss_curve.png`
- 配置文件和日志

### 3. 评估模型

训练完成后，使用以下命令评估模型性能:

```bash
python run_evaluate.py --model-path runs/模型名称_时间戳/weights/best.pt --config configs/模型名称_config.yaml
```

评估结果将保存在 `runs/模型名称_时间戳/evaluation/`目录下，包括：

- 评估指标: `evaluation_results.csv`
- 预测结果: `predictions.csv`
- 每个特征的预测vs实际散点图

### 4. 使用训练好的模型进行预测

对新数据进行预测：

```bash
python detect.py --weights runs/模型名称_时间戳/weights/best.pt --config configs/模型名称_config.yaml --input path/to/features.csv --output predictions
```

可以对单个文件或整个目录进行预测：

```bash
# 预测目录中的数据
python detect.py --weights runs/模型名称_时间戳/weights/best.pt --config configs/模型名称_config.yaml --input path/to/data_dir --output predictions

# 预测单个文件
python detect.py --weights runs/模型名称_时间戳/weights/best.pt --config configs/模型名称_config.yaml --input path/to/features.csv --output predictions
```

生成预测结果的可视化图表：

```bash
python detect.py --weights runs/模型名称_时间戳/weights/best.pt --input data/test/ --output predictions --save-plots
```

预测结果将保存在指定的 `output`目录中：

- 预测结果CSV文件：`predictions_时间戳.csv`
- 可视化图表（如果使用 `--save-plots`）：`plots/`目录下

## 模型类型

- **MLP** (多层感知机): 适用于简单数据关系
- **Conv1D** (一维卷积神经网络): 适用于序列数据和信号处理
- **LSTM** (长短期记忆网络): 适用于时序特征提取
- **Transformer**: 适用于长序列建模和注意力机制
- **MPBDNet** (多路径双分支网络): 结合卷积和循环网络，适用于复杂序列分析
- **Autoformer** (自相关变换器): 基于自相关机制的时间序列模型
- **Informer** (信息增强变换器): 使用概率注意力的高效Transformer

### MPBDNet模型

MPBDNet (Multi-Path Block with Dual branches Network) 是一个特别设计的深度神经网络，结合了卷积神经网络和循环神经网络的优势。

#### 特点

- **多路径架构**：通过并行分支提取不同尺度的特征
- **双分支块**：每个MPBDBlock包含两个并行分支，使用不同大小的卷积核
  - 第一分支：使用小卷积核(3x3)提取局部特征
  - 第二分支：使用大卷积核(5x5, 7x7)提取更广泛的特征
- **级联结构**：多个MPBDBlock串联，逐步增加通道数和抽象层次
- **双向LSTM**：捕获序列中的长期依赖关系
- **动态输入处理**：支持处理不同长度的输入序列
- **批量规范化**：可选的批归一化层，提高训练稳定性
- **特殊处理机制**：对单样本批次进行特殊处理，确保批归一化层正常工作

#### 配置示例

```yaml
# MPBDNet配置
model:
  name: mpbdnet
  input_channels: 1
  num_classes: 3
  list_inplanes: [3, 6, 18]  # 每个块的通道大小
  num_rnn_sequence: 18
  embedding_c: 50  # 嵌入维度
  seq_len: 64
  dropout_rate: 0.3
  batch_norm: true
```

### Autoformer模型

Autoformer是一种基于自相关机制的Transformer变体，专门为时间序列建模设计。

#### 特点

- **自相关机制**：利用时间序列的周期性特征，通过FFT计算序列间的自相关性
- **序列分解**：将序列数据分解为趋势和季节性成分，分别处理
- **高效编码器**：使用标准的Transformer编码器层，但带有自相关注意力机制
- **线性复杂度**：相比于标准Transformer的二次方复杂度，具有更高的计算效率
- **长序列建模**：特别适合捕获长期依赖关系和周期模式

#### 配置示例

```yaml
# Autoformer配置
model:
  name: autoformer
  input_dim: 1684      # 输入特征维度
  output_dim: 3        # 输出维度（logg, feh, teff）
  d_model: 256         # 模型维度
  n_heads: 8           # 多头注意力头数
  e_layers: 2          # 编码器层数
  d_ff: 1024           # 前馈网络维度
  dropout_rate: 0.1    # Dropout概率
  activation: gelu     # 激活函数
```

### Informer模型

Informer是一种高效的长序列Transformer变体，通过概率注意力机制降低计算复杂度。

#### 特点

- **概率稀疏注意力**：选择最具代表性的查询键值对，大幅降低计算开销
- **高效自注意力蒸馏**：通过层次化的设计减少序列长度
- **生成解码器**：使用一个步骤完成长序列预测
- **适用于长序列**：相比标准Transformer，可以处理更长的序列输入
- **内存高效**：显著降低内存使用量，使大规模序列建模成为可能

#### 配置示例

```yaml
# Informer配置
model:
  name: informer
  input_dim: 1684       # 输入特征维度
  output_dim: 3         # 输出维度（logg, feh, teff）
  d_model: 256          # 模型维度
  n_heads: 8            # 多头注意力头数
  e_layers: 3           # 编码器层数
  d_ff: 1024            # 前馈网络维度
  dropout_rate: 0.1     # Dropout概率
  activation: gelu      # 激活函数
```

## 配置文件说明

配置文件(configs/config.yaml)允许自定义模型结构和训练参数：

```yaml
# 模型配置
model:
  name: mlp  # 可选: mlp, conv1d, lstm, transformer, mpbdnet, autoformer, informer
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

- **2025-03-31**:
  - 添加了配置文件中的特征选项
  - 实现从实际数据集加载数据
  - 添加评估脚本和可视化功能
  - 支持通过obsid关联特征和标签文件
- **最新更新**:
  - 添加了MPBDNet模型，提供更强大的序列特征提取能力
  - 添加了Autoformer和Informer模型，增强了时间序列建模能力
  - 修复了detect.py脚本，使其能够从实际数据集加载数据
  - 支持使用obsid列进行预测结果输出
  - 更新了预测可视化功能，支持多特征预测结果展示
  - 改进了项目文档，添加了详细的使用说明
  - 更新了requirements.txt，添加了所有必要的依赖项
  - 将进度条显示从tqdm更换为progressbar2，提供更清晰的训练进度显示

## 使用TensorBoard监控训练过程

本框架支持使用TensorBoard监控训练过程，训练日志会自动保存在实验目录下的 `logs`文件夹内。

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
