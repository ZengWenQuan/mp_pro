#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估脚本
用法:
    python evaluate.py --model-path runs/mlp_20230330_120000/weights/best.pt --config configs/config.yaml
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from tqdm import tqdm

# 设置matplotlib字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from utils.general import seed_everything, load_config, get_device
from utils.dataset import Normalizer
from models.model import MLP, Conv1D, LSTM, SpectralTransformer


def parse_args():
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='结果输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()


def get_model(model_cfg):
    """获取模型"""
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
    
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")
    
    # 打印模型结构概要
    print(f"模型结构总结:")
    print(f"  参数总数: {sum(p.numel() for p in model.parameters())}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model


def load_data(data_cfg, dataset_type='test'):
    """加载数据集
    
    Args:
        data_cfg: 数据配置
        dataset_type: 数据集类型，'train'或'test'
    """
    # 获取ID列名
    id_column = data_cfg.get('id_column', 'obsid')
    
    # 获取要使用的标签列表
    labels = data_cfg.get('labels', ['logg', 'feh', 'teff'])
    use_all_features = data_cfg.get('use_all_features', True)
    
    # 从配置中获取数据路径
    data_dir = data_cfg.get(f'{dataset_type}_dir', f'data/{dataset_type}')
    features_file = data_cfg.get('features_file', 'features.csv')
    labels_file = data_cfg.get('labels_file', 'labels.csv')
    
    # 构建完整的文件路径
    features_path = os.path.join(data_dir, features_file)
    labels_path = os.path.join(data_dir, labels_file)
    
    print(f"加载{dataset_type}数据集:")
    print(f"  特征文件: {features_path}")
    print(f"  标签文件: {labels_path}")
    
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    
    # 根据obsid列合并特征和标签
    data = pd.merge(features_df, labels_df, on=id_column)
    
    # 提取特征
    if use_all_features:
        # 使用除id列外的所有特征列
        feature_columns = [col for col in features_df.columns if col != id_column]
    else:
        # 只使用指定的特征列
        feature_columns = labels
    
    # 获取标签列
    label_columns = labels
    
    print(f"使用特征: {feature_columns}")
    print(f"使用标签: {label_columns}")
    
    # 提取数据集的特征和标签
    X = data[feature_columns].values
    y = data[label_columns].values
    
    print(f"{dataset_type.capitalize()}数据形状: {X.shape}")
    print(f"{dataset_type.capitalize()}标签形状: {y.shape}")
    
    return X, y, data, feature_columns, label_columns


def evaluate_model(model, X_test, y_test, device, output_dir, label_columns, label_normalizer=None):
    """评估模型性能"""
    model.eval()
    model.to(device)
    
    # 批量处理测试数据，避免CUDA内存不足
    batch_size = 128  # 可根据GPU内存调整
    num_samples = X_test.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    
    y_pred_norm_list = []
    
    # 分批次处理预测
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            # 获取当前批次数据
            X_batch = X_test[start_idx:end_idx]
            X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
            
            # 打印当前批次形状（调试用）
            if i == 0:
                print(f"测试数据批次形状: {X_batch.shape}，张量形状: {X_batch_tensor.shape}")
            
            # 预测当前批次
            try:
                batch_pred = model(X_batch_tensor).cpu().numpy()
                y_pred_norm_list.append(batch_pred)
            except RuntimeError as e:
                print(f"批次 {i+1}/{num_batches} 处理错误: {e}")
                print(f"批次形状: {X_batch.shape}")
                # 如果是GPU内存错误，尝试使用CPU
                if "CUDA out of memory" in str(e):
                    print("尝试使用CPU进行此批次预测...")
                    model.to('cpu')
                    X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32)
                    batch_pred = model(X_batch_tensor).numpy()
                    y_pred_norm_list.append(batch_pred)
                    model.to(device)  # 预测完再移回GPU
                else:
                    raise
    
    # 合并所有批次的预测结果
    y_pred_norm = np.vstack(y_pred_norm_list)
    
    # 输出原始归一化预测的范围，用于调试
    print(f"归一化预测结果范围: {np.min(y_pred_norm, axis=0)} - {np.max(y_pred_norm, axis=0)}")
    
    # 如果有标签归一化器，将预测值转换回原始尺度
    if label_normalizer is not None:
        y_pred = label_normalizer.inverse_transform(y_pred_norm)
        print(f"反归一化后预测结果范围: {np.min(y_pred, axis=0)} - {np.max(y_pred, axis=0)}")
    else:
        y_pred = y_pred_norm
        print("警告: 未使用标签归一化器进行反向转换!")
    
    # 输出实际标签的范围，用于对比
    print(f"实际标签范围: {np.min(y_test, axis=0)} - {np.max(y_test, axis=0)}")
    
    # 计算评估指标
    results = {}
    for i, label in enumerate(label_columns):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        print(f"==== 特征 {label} 评估结果 ====")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print()
        
        results[label] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # 绘制预测vs实际值散点图
        plt.figure(figsize=(12, 10))
        
        # 散点图
        plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.5, label='预测点')
        
        # 添加理想线（y=x线）
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想线 (y=x)')
        
        # 计算并绘制拟合线，使用numpy的polyfit
        z = np.polyfit(y_test[:, i], y_pred[:, i], 1)
        p = np.poly1d(z)
        plt.plot(np.sort(y_test[:, i]), p(np.sort(y_test[:, i])), 'g-', lw=2, 
                 label=f'拟合线 (y={z[0]:.2f}x+{z[1]:.2f})')
        
        # 设置图表属性
        plt.xlabel('实际值', fontsize=14)
        plt.ylabel('预测值', fontsize=14)
        plt.title(f'{label} - 预测 vs 实际', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # 添加R²值文本
        plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{label}_predictions.png'), dpi=300)
        plt.close()
        
        # 保存每个标签的实际值和预测值到CSV
        result_df = pd.DataFrame({
            'actual': y_test[:, i],
            'predicted': y_pred[:, i],
            'error': y_pred[:, i] - y_test[:, i]
        })
        result_df.to_csv(os.path.join(output_dir, f'{label}_predictions_data.csv'), index=False)
    
    # 计算总体评估指标
    total_mse = mean_squared_error(y_test, y_pred)
    total_rmse = np.sqrt(total_mse)
    total_mae = mean_absolute_error(y_test, y_pred)
    
    print("==== 总体评估结果 ====")
    print(f"MSE: {total_mse:.4f}")
    print(f"RMSE: {total_rmse:.4f}")
    print(f"MAE: {total_mae:.4f}")
    
    results['overall'] = {
        'mse': total_mse,
        'rmse': total_rmse,
        'mae': total_mae
    }
    
    # 保存评估结果
    result_df = pd.DataFrame()
    for feature, metrics in results.items():
        temp_df = pd.DataFrame(metrics, index=[feature])
        result_df = pd.concat([result_df, temp_df])
    
    result_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'))
    print(f"评估结果已保存到: {os.path.join(output_dir, 'evaluation_results.csv')}")
    
    # 保存原始预测值和归一化的预测值（如果有归一化）
    if label_normalizer is not None:
        pred_comparison = pd.DataFrame()
        for i, label in enumerate(label_columns):
            pred_comparison[f'{label}_norm'] = y_pred_norm[:, i]
            pred_comparison[f'{label}_orig'] = y_pred[:, i]
        pred_comparison.to_csv(os.path.join(output_dir, 'predictions_comparison.csv'), index=False)
        print(f"预测值比较已保存到: {os.path.join(output_dir, 'predictions_comparison.csv')}")
    
    return y_pred, results


def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    seed_everything(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建主输出目录
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建训练集和测试集的子目录
    train_output_dir = output_dir / 'train'
    test_output_dir = output_dir / 'test'
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 加载模型checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 检查checkpoint中是否包含模型配置信息
    if 'model_config' in checkpoint:
        # 使用checkpoint中的模型配置
        model_config = checkpoint['model_config']
        print("使用checkpoint中的模型配置")
    else:
        # 使用配置文件中的模型配置
        model_config = config['model']
        print("使用配置文件中的模型配置")
    
    # 查找归一化参数文件
    model_dir = Path(args.model_path).parent.parent
    norm_dir = model_dir / 'normalization'
    
    # 加载标签归一化器
    label_normalizer = None
    if os.path.exists(norm_dir / 'label_normalizer.json'):
        label_normalizer = Normalizer()
        label_normalizer.load(norm_dir / 'label_normalizer.json')
        print("已加载标签归一化参数")
        
        # 输出标签归一化参数，便于调试
        print("\n标签归一化参数:")
        for k, v in label_normalizer.params.items():
            print(f"  {k}: {v}")
    
    # 加载特征归一化器
    feature_normalizer = None
    if os.path.exists(norm_dir / 'feature_normalizer.json'):
        feature_normalizer = Normalizer()
        feature_normalizer.load(norm_dir / 'feature_normalizer.json')
        print("已加载特征归一化参数")
    
    # 创建模型 - 确保使用与训练时相同的模型类型和结构
    model = None  # 初始化为None，后面会重新创建模型
    
    # 评估训练集
    print("\n=== 评估训练集 ===")
    # 加载训练数据
    X_train, y_train, train_data, feature_columns, label_columns = load_data(config['data'], 'train')
    
    # 更新模型配置中的输入维度和输出维度
    model_config['input_dim'] = X_train.shape[1]
    model_config['output_dim'] = len(label_columns)
    
    # 检查是否有保存的列名定义
    if os.path.exists(norm_dir / 'columns.json'):
        with open(norm_dir / 'columns.json', 'r') as f:
            columns = json.load(f)
            saved_label_columns = columns.get('output_label_columns', columns.get('label_columns', None))
            if saved_label_columns and set(saved_label_columns) == set(label_columns):
                print(f"验证: 使用的标签列与保存的标签列匹配")
            elif saved_label_columns:
                print(f"警告: 当前标签列 {label_columns} 与保存的标签列 {saved_label_columns} 不匹配")
    
    # 应用特征归一化
    if feature_normalizer is not None:
        X_train = feature_normalizer.transform(X_train)
    
    # 创建模型
    model = get_model(model_config)
    
    # 打印模型结构和参数数量
    print(f"\n模型结构:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 尝试加载模型权重
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功，来自: {args.model_path}")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        print("尝试使用严格=False的加载方式...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("模型权重已加载（非严格模式）")
    
    # 评估模型在训练集上的表现
    y_train_pred, train_results = evaluate_model(model, X_train, y_train, device, train_output_dir, label_columns, label_normalizer)
    
    # 保存训练集的预测结果
    train_pred_df = pd.DataFrame(y_train_pred, columns=label_columns)
    train_pred_df['obsid'] = train_data['obsid'].values
    train_pred_df.to_csv(os.path.join(train_output_dir, 'predictions.csv'), index=False)
    print(f"训练集预测结果已保存到: {os.path.join(train_output_dir, 'predictions.csv')}")
    
    # 添加更详细的训练集分析 - 生成数据分布图
    print("\n生成训练集数据分布直方图...")
    for i, label in enumerate(label_columns):
        plt.figure(figsize=(12, 8))
        
        # 创建2x1网格
        plt.subplot(2, 1, 1)
        plt.hist(y_train[:, i], bins=50, alpha=0.7, label='实际值')
        plt.axvline(np.mean(y_train[:, i]), color='r', linestyle='dashed', linewidth=1, label=f'均值: {np.mean(y_train[:, i]):.4f}')
        plt.title(f'{label} - 训练集实际值分布', fontsize=14)
        plt.xlabel('值', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.hist(y_train_pred[:, i], bins=50, alpha=0.7, label='预测值')
        plt.axvline(np.mean(y_train_pred[:, i]), color='r', linestyle='dashed', linewidth=1, label=f'均值: {np.mean(y_train_pred[:, i]):.4f}')
        plt.title(f'{label} - 训练集预测值分布', fontsize=14)
        plt.xlabel('值', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(train_output_dir, f'{label}_distribution.png'), dpi=300)
        plt.close()
    
    # 打印训练集预测值的统计信息
    print("\n训练集预测值统计:")
    for i, label in enumerate(label_columns):
        print(f"  {label}:")
        print(f"    Min: {np.min(y_train_pred[:, i]):.4f}")
        print(f"    Max: {np.max(y_train_pred[:, i]):.4f}")
        print(f"    Mean: {np.mean(y_train_pred[:, i]):.4f}")
        print(f"    Std: {np.std(y_train_pred[:, i]):.4f}")
        
        # 计算预测值的相对变化范围
        pred_range = np.max(y_train_pred[:, i]) - np.min(y_train_pred[:, i])
        actual_range = np.max(y_train[:, i]) - np.min(y_train[:, i])
        range_ratio = pred_range / actual_range if actual_range != 0 else 0
        print(f"    预测范围: {pred_range:.4f} (实际范围的 {range_ratio:.2%})")
        
        # 检查预测分布
        if np.std(y_train_pred[:, i]) < 0.01 * np.std(y_train[:, i]):
            print(f"    警告: {label}的训练集预测分布过于集中，标准差比实际小了100倍以上!")
        elif np.std(y_train_pred[:, i]) < 0.1 * np.std(y_train[:, i]):
            print(f"    警告: {label}的训练集预测分布较为集中，标准差比实际小了10倍以上!")
    
    # 评估测试集
    print("\n=== 评估测试集 ===")
    # 加载测试数据
    X_test, y_test, test_data, feature_columns, label_columns = load_data(config['data'], 'test')
    
    # 应用特征归一化
    if feature_normalizer is not None:
        X_test = feature_normalizer.transform(X_test)
    
    # 评估模型在测试集上的表现
    y_test_pred, test_results = evaluate_model(model, X_test, y_test, device, test_output_dir, label_columns, label_normalizer)
    
    # 保存测试集的预测结果
    test_pred_df = pd.DataFrame(y_test_pred, columns=label_columns)
    test_pred_df['obsid'] = test_data['obsid'].values
    test_pred_df.to_csv(os.path.join(test_output_dir, 'predictions.csv'), index=False)
    print(f"测试集预测结果已保存到: {os.path.join(test_output_dir, 'predictions.csv')}")
    
    # 添加更详细的测试集分析 - 生成数据分布图
    print("\n生成测试集数据分布直方图...")
    for i, label in enumerate(label_columns):
        plt.figure(figsize=(12, 8))
        
        # 创建2x1网格
        plt.subplot(2, 1, 1)
        plt.hist(y_test[:, i], bins=50, alpha=0.7, label='实际值')
        plt.axvline(np.mean(y_test[:, i]), color='r', linestyle='dashed', linewidth=1, label=f'均值: {np.mean(y_test[:, i]):.4f}')
        plt.title(f'{label} - 测试集实际值分布', fontsize=14)
        plt.xlabel('值', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.hist(y_test_pred[:, i], bins=50, alpha=0.7, label='预测值')
        plt.axvline(np.mean(y_test_pred[:, i]), color='r', linestyle='dashed', linewidth=1, label=f'均值: {np.mean(y_test_pred[:, i]):.4f}')
        plt.title(f'{label} - 测试集预测值分布', fontsize=14)
        plt.xlabel('值', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(test_output_dir, f'{label}_distribution.png'), dpi=300)
        plt.close()
    
    # 打印测试集预测值的统计信息
    print("\n测试集预测值统计:")
    for i, label in enumerate(label_columns):
        print(f"  {label}:")
        print(f"    Min: {np.min(y_test_pred[:, i]):.4f}")
        print(f"    Max: {np.max(y_test_pred[:, i]):.4f}")
        print(f"    Mean: {np.mean(y_test_pred[:, i]):.4f}")
        print(f"    Std: {np.std(y_test_pred[:, i]):.4f}")
        
        # 计算预测值的相对变化范围
        pred_range = np.max(y_test_pred[:, i]) - np.min(y_test_pred[:, i])
        actual_range = np.max(y_test[:, i]) - np.min(y_test[:, i])
        range_ratio = pred_range / actual_range if actual_range != 0 else 0
        print(f"    预测范围: {pred_range:.4f} (实际范围的 {range_ratio:.2%})")
        
        # 检查预测分布
        if np.std(y_test_pred[:, i]) < 0.01 * np.std(y_test[:, i]):
            print(f"    警告: {label}的测试集预测分布过于集中，标准差比实际小了100倍以上!")
        elif np.std(y_test_pred[:, i]) < 0.1 * np.std(y_test[:, i]):
            print(f"    警告: {label}的测试集预测分布较为集中，标准差比实际小了10倍以上!")
    
    # 比较训练集和测试集的评估结果
    print("\n=== 训练集与测试集评估结果比较 ===")
    comparison_df = pd.DataFrame()
    
    for label in label_columns:
        train_metrics = train_results[label]
        test_metrics = test_results[label]
        
        label_df = pd.DataFrame({
            'train_mse': [train_metrics['mse']],
            'test_mse': [test_metrics['mse']],
            'train_rmse': [train_metrics['rmse']],
            'test_rmse': [test_metrics['rmse']],
            'train_mae': [train_metrics['mae']],
            'test_mae': [test_metrics['mae']],
            'train_r2': [train_metrics['r2']],
            'test_r2': [test_metrics['r2']]
        }, index=[label])
        
        comparison_df = pd.concat([comparison_df, label_df])
        
        print(f"标签: {label}")
        print(f"  训练集 MSE: {train_metrics['mse']:.4f}, 测试集 MSE: {test_metrics['mse']:.4f}, 差异: {test_metrics['mse'] - train_metrics['mse']:.4f}")
        print(f"  训练集 RMSE: {train_metrics['rmse']:.4f}, 测试集 RMSE: {test_metrics['rmse']:.4f}, 差异: {test_metrics['rmse'] - train_metrics['rmse']:.4f}")
        print(f"  训练集 MAE: {train_metrics['mae']:.4f}, 测试集 MAE: {test_metrics['mae']:.4f}, 差异: {test_metrics['mae'] - train_metrics['mae']:.4f}")
        print(f"  训练集 R²: {train_metrics['r2']:.4f}, 测试集 R²: {test_metrics['r2']:.4f}, 差异: {test_metrics['r2'] - train_metrics['r2']:.4f}")
        print()
    
    # 添加总体评估指标比较
    train_overall = train_results['overall']
    test_overall = test_results['overall']
    
    overall_df = pd.DataFrame({
        'train_mse': [train_overall['mse']],
        'test_mse': [test_overall['mse']],
        'train_rmse': [train_overall['rmse']],
        'test_rmse': [test_overall['rmse']],
        'train_mae': [train_overall['mae']],
        'test_mae': [test_overall['mae']]
    }, index=['overall'])
    
    comparison_df = pd.concat([comparison_df, overall_df])
    
    comparison_df.to_csv(os.path.join(output_dir, 'train_test_comparison.csv'))
    print(f"训练集与测试集比较结果已保存到: {os.path.join(output_dir, 'train_test_comparison.csv')}")
    
    # 绘制训练集与测试集对比图表
    for i, label in enumerate(label_columns):
        plt.figure(figsize=(15, 10))
        
        # 创建训练集与测试集预测对比图
        plt.subplot(2, 2, 1)
        # 训练集散点图
        plt.scatter(y_train[:, i], y_train_pred[:, i], alpha=0.5, color='blue', label='训练集')
        # 测试集散点图
        plt.scatter(y_test[:, i], y_test_pred[:, i], alpha=0.5, color='red', label='测试集')
        # 理想线
        min_val = min(min(y_train[:, i].min(), y_train_pred[:, i].min()), 
                      min(y_test[:, i].min(), y_test_pred[:, i].min()))
        max_val = max(max(y_train[:, i].max(), y_train_pred[:, i].max()), 
                      max(y_test[:, i].max(), y_test_pred[:, i].max()))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        plt.title(f'{label} - 预测值 vs 实际值', fontsize=14)
        plt.xlabel('实际值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # 训练集与测试集误差分布对比
        plt.subplot(2, 2, 2)
        train_errors = y_train_pred[:, i] - y_train[:, i]
        test_errors = y_test_pred[:, i] - y_test[:, i]
        bins = np.linspace(
            min(train_errors.min(), test_errors.min()), 
            max(train_errors.max(), test_errors.max()), 
            50
        )
        plt.hist(train_errors, bins=bins, alpha=0.5, color='blue', label='训练集')
        plt.hist(test_errors, bins=bins, alpha=0.5, color='red', label='测试集')
        plt.axvline(0, color='k', linestyle='--', lw=1)
        plt.title(f'{label} - 预测误差分布', fontsize=14)
        plt.xlabel('预测误差 (预测值 - 实际值)', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # R²值对比
        plt.subplot(2, 2, 3)
        r2_vals = [train_results[label]['r2'], test_results[label]['r2']]
        bars = plt.bar(['训练集', '测试集'], r2_vals, color=['blue', 'red'])
        plt.title(f'{label} - R² 值对比', fontsize=14)
        plt.ylabel('R²', fontsize=12)
        plt.ylim(0, 1.0 if max(r2_vals) < 1.0 else 1.1)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=12)
        
        # MSE和RMSE对比
        plt.subplot(2, 2, 4)
        metrics = ['MSE', 'RMSE', 'MAE']
        train_metrics_vals = [
            train_results[label]['mse'],
            train_results[label]['rmse'],
            train_results[label]['mae']
        ]
        test_metrics_vals = [
            test_results[label]['mse'],
            test_results[label]['rmse'],
            test_results[label]['mae']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        plt.bar(x - width/2, train_metrics_vals, width, label='训练集', color='blue')
        plt.bar(x + width/2, test_metrics_vals, width, label='测试集', color='red')
        plt.title(f'{label} - 评估指标对比', fontsize=14)
        plt.xticks(x, metrics)
        plt.ylabel('值', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{label}_train_test_comparison.png'), dpi=300)
        plt.close()
    
    print("\n评估完成。")


if __name__ == '__main__':
    main() 