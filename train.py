import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import random
import time
import json
from tqdm import tqdm

from utils.general import seed_everything, load_config, create_exp_dir, get_device
from utils.dataset import create_dataloaders, Normalizer
from train.trainer import Trainer
from models.model import MLP, Conv1D, LSTM, SpectralTransformer, MPBDNet


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for prediction model')
    parser.add_argument('--config', type=str, default='configs/lstm.yaml',
                        help='Path to config file')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: model_name + timestamp)')
    return parser.parse_args()


def get_model(model_cfg):
    """Get model based on configuration"""
    model_name = model_cfg.get('name', 'mlp').lower()
    
    # 获取输入和输出维度
    input_dim = model_cfg.get('input_dim', 64)
    output_dim = model_cfg.get('output_dim', 3)
    
    print(f"创建{model_name}模型，输入维度: {input_dim}, 输出维度: {output_dim}")
    
    if model_name == 'mlp':
        model = MLP(
            input_dim=input_dim,
            hidden_dims=model_cfg.get('hidden_dims', [128, 256, 128]),
            output_dim=output_dim,
            dropout_rate=model_cfg.get('dropout_rate', 0.2)
        )
    
    elif model_name == 'conv1d':
        # 对于Conv1D模型，输入通道数始终为1，将特征数量作为序列长度
        model = Conv1D(
            input_channels=1,  # 固定为1，特征将在forward中重新排列
            seq_len=input_dim,  # 使用输入维度作为序列长度
            num_classes=output_dim,
            channels=model_cfg.get('channels', [32, 64, 128, 256, 512]),
            kernel_sizes=model_cfg.get('kernel_sizes', [3, 3, 3, 3, 3]),
            fc_dims=model_cfg.get('fc_dims', [512, 256]),
            dropout_rate=model_cfg.get('dropout_rate', 0.2)
        )
    
    elif model_name == 'lstm':
        # 对于LSTM模型，我们假设输入特征维度为1（单通道），使用config.seq_len作为序列长度
        model = LSTM(
            input_dim=1,  # 固定为1，一般光谱数据每个时间点一个特征值
            hidden_dim=model_cfg.get('hidden_dim', 128),
            num_layers=model_cfg.get('num_layers', 2),
            bidirectional=model_cfg.get('bidirectional', True),
            dropout_rate=model_cfg.get('dropout_rate', 0.2),
            output_dim=output_dim
        )
    
    elif model_name == 'transformer':
        # 对于Transformer模型，我们假设输入特征维度为1，使用config.seq_len作为序列长度
        model = SpectralTransformer(
            input_dim=1,  # 固定为1，光谱数据每个时间点一个特征值
            d_model=model_cfg.get('d_model', 128),
            nhead=model_cfg.get('nhead', 8),
            num_layers=model_cfg.get('num_layers', 3),
            dim_feedforward=model_cfg.get('dim_feedforward', 512),
            dropout_rate=model_cfg.get('dropout_rate', 0.1),
            output_dim=output_dim
        )
    
    elif model_name == 'mpbdnet':
        # 对于MPBDNet模型，我们使用配置中的参数
        model = MPBDNet(
            num_classes=output_dim,
            list_inplanes=model_cfg.get('list_inplanes', [3, 6, 18]),
            num_rnn_sequence=model_cfg.get('num_rnn_sequence', 18),
            embedding_c=model_cfg.get('embedding_c', 50),
            seq_len=input_dim,
            dropout_rate=model_cfg.get('dropout_rate', 0.3)
        )
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # 打印模型结构概要
    print(f"模型结构总结:")
    print(f"  参数总数: {sum(p.numel() for p in model.parameters())}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model


def load_data(data_cfg):
    """Load data from actual dataset"""
    # 获取ID列名
    id_column = data_cfg.get('id_column', 'obsid')
    
    # 获取要使用的标签列表
    labels = data_cfg.get('labels', ['logg', 'feh', 'teff'])
    use_all_features = data_cfg.get('use_all_features', True)
    
    # 从配置中获取数据路径
    train_dir = data_cfg.get('train_dir', 'data/train')
    test_dir = data_cfg.get('test_dir', 'data/test')
    features_file = data_cfg.get('features_file', 'features.csv')
    labels_file = data_cfg.get('labels_file', 'labels.csv')
    
    # 构建完整的文件路径
    train_features_path = os.path.join(train_dir, features_file)
    train_labels_path = os.path.join(train_dir, labels_file)
    test_features_path = os.path.join(test_dir, features_file)
    test_labels_path = os.path.join(test_dir, labels_file)
    
    # 加载训练集
    train_features = pd.read_csv(train_features_path)
    train_labels = pd.read_csv(train_labels_path)
    
    # 加载测试集
    test_features = pd.read_csv(test_features_path)
    test_labels = pd.read_csv(test_labels_path)
    
    # 根据obsid列合并特征和标签
    train_data = pd.merge(train_features, train_labels, on=id_column)
    test_data = pd.merge(test_features, test_labels, on=id_column)
    
    # 提取特征
    if use_all_features:
        # 使用除id列外的所有特征列
        feature_columns = [col for col in train_features.columns if col != id_column]
    else:
        # 只使用指定的特征列
        feature_columns = labels
    
    # 获取标签列
    label_columns = labels
    
    # 只输出标签名，不输出特征名
    print(f"Using labels: {label_columns}")
    
    # 提取训练集和测试集的特征和标签
    X_train = train_data[feature_columns].values
    y_train = train_data[label_columns].values
    
    X_test = test_data[feature_columns].values
    y_test = test_data[label_columns].values
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # 返回原始数据，不进行归一化，归一化将在main函数中进行
    return X_train, y_train, X_test, y_test, feature_columns, label_columns, test_data[id_column].values


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create experiment directory
    if args.exp_name:
        exp_name = args.exp_name
    else:
        model_name = config['model']['name']
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        exp_name = f"{model_name}_{timestamp}"
    
    exp_dir = create_exp_dir(exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Update config with command line arguments
    if args.resume_from:
        config['resume_from'] = args.resume_from
    
    # Save config to experiment directory
    config_path = exp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config, f)
    
    # Load data
    X_train, y_train, X_test, y_test, feature_columns, label_columns, test_ids = load_data(config['data'])
    
    # 打印原始标签数据的统计信息
    print("\n原始标签数据统计:")
    for i, col in enumerate(label_columns):
        print(f"  {col}:")
        print(f"    Min: {np.min(y_train[:, i])}")
        print(f"    Max: {np.max(y_train[:, i])}")
        print(f"    Mean: {np.mean(y_train[:, i])}")
        print(f"    Std: {np.std(y_train[:, i])}")
    
    # 创建归一化器并进行特征归一化
    feature_normalizer = Normalizer(method='standard')
    X_train_norm = feature_normalizer.fit_transform(X_train)
    X_test_norm = feature_normalizer.transform(X_test)
    
    # 保存特征归一化参数
    norm_dir = Path(exp_dir) / 'normalization'
    os.makedirs(norm_dir, exist_ok=True)
    feature_normalizer.save(norm_dir / 'feature_normalizer.json')
    
    # 打印特征归一化参数
    print("\n特征归一化参数:")
    for k, v in feature_normalizer.params.items():
        if len(v) > 10:  # 如果参数太长，只显示前几个
            print(f"  {k}: {v[:5]}... (长度: {len(v)})")
        else:
            print(f"  {k}: {v}")
    
    # 标签归一化
    label_normalizer = Normalizer(method='standard')
    y_train_norm = label_normalizer.fit_transform(y_train)
    y_test_norm = label_normalizer.transform(y_test)
    
    # 保存标签归一化参数
    label_normalizer.save(norm_dir / 'label_normalizer.json')
    
    # 打印标签归一化参数
    print("\n标签归一化参数:")
    for k, v in label_normalizer.params.items():
        print(f"  {k}: {v}")
    
    # 打印归一化后的标签数据统计信息
    print("\n归一化后标签数据统计:")
    for i, col in enumerate(label_columns):
        print(f"  {col}:")
        print(f"    Min: {np.min(y_train_norm[:, i])}")
        print(f"    Max: {np.max(y_train_norm[:, i])}")
        print(f"    Mean: {np.mean(y_train_norm[:, i])}")
        print(f"    Std: {np.std(y_train_norm[:, i])}")
    
    # 创建归一化前后的标签值比较表，用于调试
    y_compare = pd.DataFrame()
    for i, col in enumerate(label_columns):
        y_compare[f'{col}_orig'] = y_train[:20, i]
        y_compare[f'{col}_norm'] = y_train_norm[:20, i]
    y_compare.to_csv(norm_dir / 'label_normalization_example.csv', index=False)
    print(f"已保存标签归一化示例到: {norm_dir / 'label_normalization_example.csv'}")
    
    # 更新模型配置中的输入维度
    config['model']['input_dim'] = X_train.shape[1]
    config['model']['output_dim'] = y_train.shape[1]
    
    # Create data loaders - 使用归一化后的数据
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_norm, 
        y_train_norm,
        batch_size=config['training'].get('batch_size', 32),
        val_split=config['data'].get('val_split', 0.2),
        test_split=0,  # 不再从训练集分割测试集，而是使用单独的测试集
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # 创建单独的测试集数据加载器 - 使用归一化后的数据
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(
        torch.tensor(X_test_norm, dtype=torch.float32),
        torch.tensor(y_test_norm, dtype=torch.float32)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training'].get('batch_size', 32),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # Create model
    model = get_model(config['model'])
    print(f"Model: {model}")
    
    # 如果指定了断点路径，从断点恢复训练
    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 打印模型输出层
    if isinstance(model, MLP):
        print(f"MLP输出层: {model.model[-1]}")
    elif isinstance(model, Conv1D):
        print(f"Conv1D输出层: {model.fc_layers[-1]}")
    elif isinstance(model, MPBDNet):
        print(f"MPBDNet输出层: {model.output}")
    
    # 将模型配置添加到训练配置中
    config['training']['model_config'] = config['model']
    
    # 添加配置文件路径到训练配置中，用于模型评估
    config['training']['config_path'] = args.config
    
    # 保存特征列和标签列以供后续使用
    with open(norm_dir / 'columns.json', 'w') as f:
        json.dump({
            'input_feature_columns': feature_columns,
            'output_label_columns': label_columns
        }, f)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        experiment_dir=exp_dir
    )
    
    # Train model
    best_val_loss = trainer.train()
    
    print(f"Training completed with best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {exp_dir}")


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    batch_losses = []
    
    # 创建进度条，设置leave=True保持显示
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs} [Train]', 
                leave=True, position=0, ncols=100)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # 更新总损失和批次损失列表
        total_loss += loss.item()
        batch_losses.append(loss.item())
        
        # 计算平均损失
        avg_loss = total_loss / (batch_idx + 1)
        
        # 更新进度条，但不重新创建
        pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    # 关闭进度条
    pbar.close()
    
    return total_loss / len(train_loader), batch_losses

def validate(model, val_loader, criterion, device, epoch, total_epochs):
    """验证模型"""
    model.eval()
    total_loss = 0
    batch_losses = []
    
    # 创建进度条，设置leave=True保持显示
    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{total_epochs} [Val]', 
                leave=True, position=1, ncols=100)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            # 更新总损失和批次损失列表
            total_loss += loss.item()
            batch_losses.append(loss.item())
            
            # 计算平均损失
            avg_loss = total_loss / (batch_idx + 1)
            
            # 更新进度条，但不重新创建
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    # 关闭进度条
    pbar.close()
    
    return total_loss / len(val_loader), batch_losses


if __name__ == '__main__':
    main() 