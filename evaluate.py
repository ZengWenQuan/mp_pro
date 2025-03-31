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

# 设置matplotlib字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from utils.general import seed_everything, load_config, get_device
from utils.dataset import Normalizer
from models import MLP, Conv1D


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
        raise ValueError(f"不支持的模型类型: {model_name}")
    
    return model


def load_data(data_cfg):
    """加载测试数据"""
    # 获取ID列名
    id_column = data_cfg.get('id_column', 'obsid')
    
    # 获取要使用的特征列表
    features = data_cfg.get('features', ['logg', 'feh', 'teff'])
    use_all_features = data_cfg.get('use_all_features', True)
    
    # 加载测试集
    test_features_path = 'data/test/features.csv'
    test_labels_path = 'data/test/labels.csv'
    
    test_features = pd.read_csv(test_features_path)
    test_labels = pd.read_csv(test_labels_path)
    
    # 根据obsid列合并特征和标签
    test_data = pd.merge(test_features, test_labels, on=id_column)
    
    # 提取特征
    if use_all_features:
        # 使用除id列外的所有特征列
        feature_columns = [col for col in test_features.columns if col != id_column]
    else:
        # 只使用指定的特征列
        feature_columns = features
    
    # 获取标签列
    label_columns = features
    
    print(f"使用特征: {feature_columns}")
    print(f"使用标签: {label_columns}")
    
    # 提取测试集的特征和标签
    X_test = test_data[feature_columns].values
    y_test = test_data[label_columns].values
    
    print(f"测试数据形状: {X_test.shape}")
    print(f"测试标签形状: {y_test.shape}")
    
    return X_test, y_test, test_data, feature_columns, label_columns


def evaluate_model(model, X_test, y_test, device, output_dir, label_columns, label_normalizer=None):
    """评估模型性能"""
    model.eval()
    model.to(device)
    
    # 转换为torch张量
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # 预测
    with torch.no_grad():
        y_pred_norm = model(X_test_tensor).cpu().numpy()
    
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
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 加载数据
    X_test, y_test, test_data, feature_columns, label_columns = load_data(config['data'])
    
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
    
    # 更新模型配置中的输入维度和输出维度
    model_config['input_dim'] = X_test.shape[1]
    model_config['output_dim'] = len(label_columns)
    
    # 查找归一化参数文件
    model_dir = Path(args.model_path).parent.parent
    norm_dir = model_dir / 'normalization'
    
    # 加载特征归一化器
    feature_normalizer = None
    if os.path.exists(norm_dir / 'feature_normalizer.json'):
        feature_normalizer = Normalizer()
        feature_normalizer.load(norm_dir / 'feature_normalizer.json')
        print("已加载特征归一化参数")
        
        # 应用特征归一化
        X_test = feature_normalizer.transform(X_test)
    
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
    
    # 创建模型 - 确保使用与训练时相同的模型类型和结构
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
    
    # 评估模型
    y_pred, results = evaluate_model(model, X_test, y_test, device, output_dir, label_columns, label_normalizer)
    
    # 保存预测结果
    pred_df = pd.DataFrame(y_pred, columns=label_columns)
    pred_df['obsid'] = test_data['obsid'].values
    pred_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print(f"预测结果已保存到: {os.path.join(output_dir, 'predictions.csv')}")
    
    # 添加更详细的分析 - 生成数据分布图
    print("\n生成数据分布直方图...")
    for i, label in enumerate(label_columns):
        plt.figure(figsize=(12, 8))
        
        # 创建2x1网格
        plt.subplot(2, 1, 1)
        plt.hist(y_test[:, i], bins=50, alpha=0.7, label='实际值')
        plt.axvline(np.mean(y_test[:, i]), color='r', linestyle='dashed', linewidth=1, label=f'均值: {np.mean(y_test[:, i]):.4f}')
        plt.title(f'{label} - 实际值分布', fontsize=14)
        plt.xlabel('值', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.hist(y_pred[:, i], bins=50, alpha=0.7, label='预测值')
        plt.axvline(np.mean(y_pred[:, i]), color='r', linestyle='dashed', linewidth=1, label=f'均值: {np.mean(y_pred[:, i]):.4f}')
        plt.title(f'{label} - 预测值分布', fontsize=14)
        plt.xlabel('值', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{label}_distribution.png'), dpi=300)
        plt.close()
    
    # 打印预测值的统计信息
    print("\n预测值统计:")
    for i, label in enumerate(label_columns):
        print(f"  {label}:")
        print(f"    Min: {np.min(y_pred[:, i]):.4f}")
        print(f"    Max: {np.max(y_pred[:, i]):.4f}")
        print(f"    Mean: {np.mean(y_pred[:, i]):.4f}")
        print(f"    Std: {np.std(y_pred[:, i]):.4f}")
        
        # 计算预测值的相对变化范围
        pred_range = np.max(y_pred[:, i]) - np.min(y_pred[:, i])
        actual_range = np.max(y_test[:, i]) - np.min(y_test[:, i])
        range_ratio = pred_range / actual_range if actual_range != 0 else 0
        print(f"    预测范围: {pred_range:.4f} (实际范围的 {range_ratio:.2%})")
        
        # 检查预测分布
        if np.std(y_pred[:, i]) < 0.01 * np.std(y_test[:, i]):
            print(f"    警告: {label}的预测分布过于集中，标准差比实际小了100倍以上!")
        elif np.std(y_pred[:, i]) < 0.1 * np.std(y_test[:, i]):
            print(f"    警告: {label}的预测分布较为集中，标准差比实际小了10倍以上!")
    
    print("\n评估完成。")


if __name__ == '__main__':
    main() 