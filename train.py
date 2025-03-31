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

from utils.general import seed_everything, load_config, create_exp_dir, get_device
from utils.dataset import create_dataloaders, Normalizer
from train.trainer import Trainer
from models import MLP, Conv1D


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for prediction model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: model_name + timestamp)')
    return parser.parse_args()


def get_model(model_cfg):
    """Get model based on configuration"""
    model_name = model_cfg.get('name', 'mlp').lower()
    
    if model_name == 'mlp':
        model = MLP(
            input_dim=model_cfg.get('input_dim', 64),
            hidden_dims=model_cfg.get('hidden_dims', [128, 256, 128]),
            output_dim=model_cfg.get('output_dim', 1),
            dropout_rate=model_cfg.get('dropout_rate', 0.2)
        )
    
    elif model_name == 'conv1d':
        # 对于Conv1D模型，输入通道数始终为1，将特征数量作为序列长度
        input_dim = model_cfg.get('input_dim', 64)
        model = Conv1D(
            input_channels=1,  # 固定为1，特征将在forward中重新排列
            seq_len=input_dim,  # 使用特征数量作为序列长度
            num_classes=model_cfg.get('output_dim', 1),
            channels=model_cfg.get('channels', [32, 64, 128]),
            kernel_sizes=model_cfg.get('kernel_sizes', [3, 3, 3]),
            fc_dims=model_cfg.get('fc_dims', [256, 128]),
            dropout_rate=model_cfg.get('dropout_rate', 0.2)
        )
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def load_data(data_cfg):
    """Load data from actual dataset"""
    # 获取ID列名
    id_column = data_cfg.get('id_column', 'obsid')
    
    # 获取要使用的标签列表
    labels = data_cfg.get('labels', ['logg', 'feh', 'teff'])
    use_all_features = data_cfg.get('use_all_features', True)
    
    # 加载训练集
    train_features_path = 'data/train/features.csv'
    train_labels_path = 'data/train/labels.csv'
    
    train_features = pd.read_csv(train_features_path)
    train_labels = pd.read_csv(train_labels_path)
    
    # 加载测试集
    test_features_path = 'data/test/features.csv'
    test_labels_path = 'data/test/labels.csv'
    
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
    if args.resume:
        config['resume'] = args.resume
    
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
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 打印模型输出层
    if isinstance(model, MLP):
        print(f"MLP输出层: {model.layers[-1]}")
    elif isinstance(model, Conv1D):
        print(f"Conv1D输出层: {model.fc_layers[-1]}")
    
    # 将模型配置添加到训练配置中
    config['training']['model_config'] = config['model']
    
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


if __name__ == '__main__':
    main() 