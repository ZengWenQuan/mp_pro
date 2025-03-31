import os
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

from models import MLP, Conv1D
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
    device = get_device()
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Determine model type and parameters from config or checkpoint
    if config is not None:
        model_cfg = config['model']
        model_name = model_cfg.get('name', 'mlp').lower()
    else:
        # Try to infer model type from checkpoint
        # This is a simplified approach; in a real application, you might
        # want to save model type and params in the checkpoint
        if 'model_config' in checkpoint and 'name' in checkpoint['model_config']:
            model_name = checkpoint['model_config']['name'].lower()
            model_cfg = checkpoint['model_config']
        else:
            state_dict = checkpoint['model_state_dict']
            keys = list(state_dict.keys())
            
            if any('conv' in key for key in keys):
                model_name = 'conv1d'
                model_cfg = {
                    'input_channels': 1,
                    'seq_len': 64,
                    'num_classes': 1,
                    'channels': [32, 64, 128],
                    'kernel_sizes': [3, 3, 3],
                    'fc_dims': [256, 128],
                    'dropout_rate': 0.0  # No dropout during inference
                }
            else:
                model_name = 'mlp'
                model_cfg = {
                    'input_dim': 64,
                    'hidden_dims': [128, 256, 128],
                    'output_dim': 1,
                    'dropout_rate': 0.0  # No dropout during inference
                }
    
    # Create model
    if model_name == 'mlp':
        model = MLP(
            input_dim=model_cfg.get('input_dim', 64),
            hidden_dims=model_cfg.get('hidden_dims', [128, 256, 128]),
            output_dim=model_cfg.get('output_dim', 1),
            dropout_rate=model_cfg.get('dropout_rate', 0.0)
        )
    elif model_name == 'conv1d':
        model = Conv1D(
            input_channels=model_cfg.get('input_channels', 1),
            seq_len=model_cfg.get('seq_len', 64),
            num_classes=model_cfg.get('num_classes', 1),
            channels=model_cfg.get('channels', [32, 64, 128]),
            kernel_sizes=model_cfg.get('kernel_sizes', [3, 3, 3]),
            fc_dims=model_cfg.get('fc_dims', [256, 128]),
            dropout_rate=model_cfg.get('dropout_rate', 0.0)
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_name} model from {weights_path}")
    return model, model_name


def load_input_data(input_path):
    """
    Load input data for prediction
    
    Args:
        input_path: Path to input data file or directory
        
    Returns:
        data: Input data as numpy array
        obsids: List of observation IDs (if available)
    """
    if os.path.isdir(input_path):
        print(f"Loading data from directory: {input_path}")
        # 如果是目录，检查是否有features.csv文件
        features_path = os.path.join(input_path, 'features.csv')
        if os.path.exists(features_path):
            df = pd.read_csv(features_path)
            print(f"Loaded {len(df)} samples from {features_path}")
            
            # 提取obsid列（如果存在）
            obsids = df['obsid'].values if 'obsid' in df.columns else None
            
            # 移除非特征列（如obsid）
            if 'obsid' in df.columns:
                features_df = df.drop(columns=['obsid'])
                data = features_df.values
            else:
                data = df.values
                
            return data, obsids
        else:
            raise FileNotFoundError(f"Could not find features.csv in {input_path}")
    else:
        print(f"Loading data from file: {input_path}")
        # 处理单个CSV文件
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
            print(f"Loaded {len(df)} samples from {input_path}")
            
            # 提取obsid列（如果存在）
            obsids = df['obsid'].values if 'obsid' in df.columns else None
            
            # 移除非特征列（如obsid）
            if 'obsid' in df.columns:
                features_df = df.drop(columns=['obsid'])
                data = features_df.values
            else:
                data = df.values
                
            return data, obsids
        else:
            raise ValueError(f"Unsupported file format: {input_path}")


def predict(model, data, model_name, batch_size=16, feature_normalizer=None, label_normalizer=None):
    """Run prediction on input data"""
    device = get_device()
    
    # Convert data to tensor
    if isinstance(data, np.ndarray):
        # 应用特征归一化
        if feature_normalizer is not None:
            data = feature_normalizer.transform(data)
        data = torch.tensor(data, dtype=torch.float32)
    
    # Prepare data based on model type
    if model_name == 'conv1d' and len(data.shape) == 2:
        # Add channel dimension for Conv1D
        data = data.unsqueeze(1)
    
    # Run prediction in batches
    normalized_predictions = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device)
            output = model(batch)
            normalized_predictions.append(output.cpu().numpy())
    
    # Concatenate batch predictions
    normalized_predictions = np.concatenate(normalized_predictions, axis=0)
    
    # 将归一化的预测转换回原始尺度
    if label_normalizer is not None:
        predictions = label_normalizer.inverse_transform(normalized_predictions)
    else:
        predictions = normalized_predictions
    
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
    data, obsids = load_input_data(args.input)
    
    # 查找并加载归一化参数
    model_dir = Path(args.weights).parent.parent
    norm_dir = model_dir / 'normalization'
    
    # 标签列
    label_columns = None
    if os.path.exists(norm_dir / 'columns.json'):
        with open(norm_dir / 'columns.json', 'r') as f:
            columns = json.load(f)
            label_columns = columns.get('label_columns', None)
    
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