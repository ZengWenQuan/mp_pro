import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import pandas as pd
import json

# 设置matplotlib字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from models.model import MLP, Conv1D, LSTM, SpectralTransformer
from utils.general import load_config, get_device
from utils.dataset import Normalizer


def parse_args():
    parser = argparse.ArgumentParser(description='Prediction script for trained model')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to model config (optional)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input data or directory')
    parser.add_argument('--output', type=str, default='predictions',
                        help='Output directory for predictions')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for prediction')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save prediction plots')
    return parser.parse_args()


def load_model(weights_path, config=None):
    """Load trained model from weights"""
    if config is None:
        print("No config provided, loading from weights file...")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
    # 获取模型配置
    model_cfg = config.get('model', {})
    model_name = model_cfg.get('name')
    
    print(f"Loading {model_name} model...")
    
    if model_name == 'mlp':
        from models.model.mlp import MLP
        model = MLP(
            input_dim=model_cfg.get('input_dim', 1),
            hidden_dims=model_cfg.get('hidden_dims', [64, 64]),
            output_dim=model_cfg.get('output_dim', 3),
            dropout_rate=model_cfg.get('dropout_rate', 0.1),
            batch_norm=model_cfg.get('batch_norm', False)
        )
    elif model_name == 'conv1d':
        from models.model.conv1d import Conv1D
        model = Conv1D(
            input_dim=model_cfg.get('input_dim', 1),
            hidden_dim=model_cfg.get('hidden_dim', 64),
            kernel_size=model_cfg.get('kernel_size', 3),
            num_layers=model_cfg.get('num_layers', 3),
            output_dim=model_cfg.get('output_dim', 3),
            dropout_rate=model_cfg.get('dropout_rate', 0.1),
            batch_norm=model_cfg.get('batch_norm', False)
        )
    elif model_name == 'lstm':
        from models.model.lstm import LSTM
        model = LSTM(
            input_dim=model_cfg.get('input_dim', 1),
            hidden_dim=model_cfg.get('hidden_dim', 64),
            num_layers=model_cfg.get('num_layers', 2),
            output_dim=model_cfg.get('output_dim', 3),
            dropout_rate=model_cfg.get('dropout_rate', 0.1),
            bidirectional=model_cfg.get('bidirectional', False),
            batch_norm=model_cfg.get('batch_norm', False)
        )
    elif model_name == 'transformer':
        from models.model.transformer import SpectralTransformer
        model = SpectralTransformer(
            input_dim=model_cfg.get('input_dim', 1),
            d_model=model_cfg.get('d_model', 128),
            nhead=model_cfg.get('nhead', 8),
            num_layers=model_cfg.get('num_layers', 3),
            dim_feedforward=model_cfg.get('dim_feedforward', 512),
            dropout_rate=model_cfg.get('dropout_rate', 0.1),
            output_dim=model_cfg.get('output_dim', 3),
            batch_norm=model_cfg.get('batch_norm', False)
        )
    elif model_name == 'mpbdnet':
        from models.model.mpbdnet import MPBDNet
        model = MPBDNet(
            input_dim=model_cfg.get('input_dim', 1),
            hidden_dim=model_cfg.get('hidden_dim', 64),
            output_dim=model_cfg.get('output_dim', 3),
            num_layers=model_cfg.get('num_layers', 3),
            dropout_rate=model_cfg.get('dropout_rate', 0.1)
        )
    elif model_name == 'autoencoder':
        from models.model.autoencoder import Autoencoder
        model = Autoencoder(
            input_dim=model_cfg.get('input_dim', 1),
            latent_dim=model_cfg.get('latent_dim', 32),
            output_dim=model_cfg.get('output_dim', 3)
        )
    elif model_name == 'autoformer':
        from models.model.autoformer import Autoformer
        model = Autoformer(
            input_dim=model_cfg.get('input_dim', 1),
            output_dim=model_cfg.get('output_dim', 3),
            d_model=model_cfg.get('d_model', 512),
            n_heads=model_cfg.get('n_heads', 8),
            e_layers=model_cfg.get('e_layers', 3),
            d_layers=model_cfg.get('d_layers', 2),
            d_ff=model_cfg.get('d_ff', 2048),
            moving_avg=model_cfg.get('moving_avg', 25),
            dropout=model_cfg.get('dropout_rate', 0.05),
            activation=model_cfg.get('activation', 'gelu'),
            output_attention=model_cfg.get('output_attention', False)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # 加载权重
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    else:
        print(f"Warning: Weights file {weights_path} not found, using untrained model")
    
    return model


def load_input_data(input_path, config=None):
    """
    Load input data for prediction
    
    Args:
        input_path: Path to input data file or directory
        config: Optional configuration dictionary
        
    Returns:
        data: Input data as numpy array
        obsids: List of observation IDs (if available)
    """
    if os.path.isdir(input_path):
        print(f"Loading data from directory: {input_path}")
        # 如果提供了配置，使用配置中的文件名
        features_file = config['data'].get('features_file', 'features.csv') if config else 'features.csv'
        features_path = os.path.join(input_path, features_file)
        
        if os.path.exists(features_path):
            df = pd.read_csv(features_path)
            print(f"Loaded {len(df)} samples from {features_path}")
            
            # 提取obsid列（如果存在）
            id_column = config['data'].get('id_column', 'obsid') if config else 'obsid'
            obsids = df[id_column].values if id_column in df.columns else None
            
            # 移除非特征列（如obsid）
            if id_column in df.columns:
                features_df = df.drop(columns=[id_column])
                data = features_df.values
            else:
                data = df.values
                
            return data, obsids
        else:
            raise FileNotFoundError(f"Could not find {features_file} in {input_path}")
    else:
        print(f"Loading data from file: {input_path}")
        # 处理单个CSV文件
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
            print(f"Loaded {len(df)} samples from {input_path}")
            
            # 提取obsid列（如果存在）
            id_column = config['data'].get('id_column', 'obsid') if config else 'obsid'
            obsids = df[id_column].values if id_column in df.columns else None
            
            # 移除非特征列（如obsid）
            if id_column in df.columns:
                features_df = df.drop(columns=[id_column])
                data = features_df.values
            else:
                data = df.values
            
            return data, obsids
        else:
            raise ValueError("Input file must be a CSV file")


def predict(model, data, model_name, batch_size=16, feature_normalizer=None, label_normalizer=None):
    """Run prediction on input data"""
    device = get_device()
    
    # 打印原始数据信息
    print(f"原始数据类型: {type(data)}, 形状: {data.shape if isinstance(data, np.ndarray) else 'unknown'}")
    
    # Convert data to tensor
    if isinstance(data, np.ndarray):
        # 应用特征归一化
        if feature_normalizer is not None:
            print(f"应用特征归一化前数据范围: {np.min(data)} - {np.max(data)}")
            data = feature_normalizer.transform(data)
            print(f"应用特征归一化后数据范围: {np.min(data)} - {np.max(data)}")
        
        # 转换为张量
        data = torch.tensor(data, dtype=torch.float32)
        print(f"转换为张量后数据形状: {data.shape}")
    
    # 打印数据统计信息
    print(f"数据统计: 最小值={data.min().item():.4f}, 最大值={data.max().item():.4f}, 均值={data.mean().item():.4f}, 标准差={data.std().item():.4f}")
    
    # 检查数据是否包含NaN或Inf
    if torch.isnan(data).any() or torch.isinf(data).any():
        print("警告: 数据中包含NaN或Inf值!")
    
    # 根据模型类型预处理数据
    if model_name == 'conv1d' and len(data.shape) == 2:
        # Add channel dimension for Conv1D
        print(f"为Conv1D模型添加通道维度，将形状从{data.shape}更改为", end=" ")
        data = data.unsqueeze(1)
        print(f"{data.shape}")
    
    # Run prediction in batches
    normalized_predictions = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device)
            try:
                output = model(batch)
                normalized_predictions.append(output.cpu().numpy())
            except RuntimeError as e:
                print(f"预测过程中出错: {e}")
                print(f"批次形状: {batch.shape}")
                raise
    
    # Concatenate batch predictions
    normalized_predictions = np.concatenate(normalized_predictions, axis=0)
    print(f"归一化预测结果形状: {normalized_predictions.shape}, 范围: {np.min(normalized_predictions)} - {np.max(normalized_predictions)}")
    
    # 将归一化的预测转换回原始尺度
    if label_normalizer is not None:
        print(f"应用标签反归一化...")
        predictions = label_normalizer.inverse_transform(normalized_predictions)
        print(f"反归一化后预测结果范围: {np.min(predictions)} - {np.max(predictions)}")
    else:
        predictions = normalized_predictions
        print("警告: 未应用标签反归一化")
    
    return predictions


def plot_prediction(data, prediction, index, label_columns=None, save_path=None):
    """Plot input data and prediction"""
    plt.figure(figsize=(12, 10))
    
    # 获取预测的特征数量
    n_features = prediction.shape[0]
    
    # 使用默认特征名称，如果没有提供
    if label_columns is None:
        if n_features == 3:
            label_columns = ['logg', 'feh', 'teff']
        else:
            label_columns = [f'Feature {i+1}' for i in range(n_features)]
    
    # 预测结果可视化
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    
    # 使用条形图显示预测结果
    plt.subplot(1, 1, 1)
    bars = plt.bar(label_columns, prediction, color=colors[:n_features], alpha=0.7)
    
    # 在条形上方显示具体预测值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prediction[i]:.4f}',
                ha='center', va='bottom', fontsize=12, rotation=0)
    
    plt.title(f"样本 {index} 的预测结果", fontsize=16)
    plt.ylabel("预测值", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 添加注释，解释标签含义
    if n_features == 3 and set(label_columns) == set(['logg', 'feh', 'teff']):
        plt.figtext(0.5, 0.01, 
                   "logg: 表面重力加速度 | feh: 金属丰度 | teff: 有效温度", 
                   ha="center", fontsize=12, 
                   bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()


def save_predictions(predictions, output_dir, obsids=None, label_columns=None):
    """Save predictions to file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    
    # 创建预测结果DataFrame
    if label_columns is None:
        # 如果没有指定标签列名，则使用默认列名
        if predictions.shape[1] == 3:
            # 假设默认预测三个特征: logg, feh, teff
            columns = ['logg', 'feh', 'teff']
        else:
            # 使用通用列名
            columns = [f'prediction_{i}' for i in range(predictions.shape[1])]
    else:
        columns = label_columns
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(predictions, columns=columns)
    
    # 添加obsid列（如果有）
    if obsids is not None:
        results_df.insert(0, 'obsid', obsids)
    else:
        # 添加一个索引列
        results_df.insert(0, 'id', range(len(predictions)))
    
    # 保存到CSV
    results_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    return output_path


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Load model
    model, model_name = load_model(args.weights, config)
    
    # Load input data
    data, obsids = load_input_data(args.input, config)
    
    # 查找并加载归一化参数
    model_dir = Path(args.weights).parent.parent
    norm_dir = model_dir / 'normalization'
    
    # 标签列
    label_columns = None
    if os.path.exists(norm_dir / 'columns.json'):
        with open(norm_dir / 'columns.json', 'r') as f:
            columns = json.load(f)
            # 使用新的键名，同时保持向后兼容性
            label_columns = columns.get('output_label_columns', columns.get('label_columns', None))
    
    # 加载特征归一化器
    feature_normalizer = None
    if os.path.exists(norm_dir / 'feature_normalizer.json'):
        feature_normalizer = Normalizer()
        feature_normalizer.load(norm_dir / 'feature_normalizer.json')
        print("已加载特征归一化参数")
    
    # 加载标签归一化器
    label_normalizer = None
    if os.path.exists(norm_dir / 'label_normalizer.json'):
        label_normalizer = Normalizer()
        label_normalizer.load(norm_dir / 'label_normalizer.json')
        print("已加载标签归一化参数")
    
    # Run prediction
    predictions = predict(model, data, model_name, args.batch_size, feature_normalizer, label_normalizer)
    
    # Save predictions
    output_dir = args.output
    output_path = save_predictions(predictions, output_dir, obsids, label_columns)
    
    # Plot and save predictions if requested
    if args.save_plots:
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        for i, (sample, pred) in enumerate(zip(data, predictions)):
            plot_path = os.path.join(plots_dir, f"prediction_{i}.png")
            plot_prediction(sample, pred, i, label_columns, save_path=plot_path)
        
        print(f"Saved prediction plots to {plots_dir}")
    
    print(f"Prediction completed. Made {len(predictions)} predictions.")


if __name__ == '__main__':
    main() 