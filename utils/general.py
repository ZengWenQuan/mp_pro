import os
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

# 设置matplotlib字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def seed_everything(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """Save configuration to YAML file"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)


def plot_loss_curve(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def create_exp_dir(exp_name, parent_dir='runs'):
    """Create experiment directory"""
    exp_dir = Path(parent_dir) / exp_name
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    weights_dir = exp_dir / 'weights'
    plots_dir = exp_dir / 'plots'
    logs_dir = exp_dir / 'logs'
    
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return exp_dir


def get_device():
    """Get available device (CUDA or CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(state, is_best, exp_dir, filename='checkpoint.pt'):
    """Save model checkpoint"""
    checkpoint_path = Path(exp_dir) / 'weights' / filename
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = Path(exp_dir) / 'weights' / 'best.pt'
        torch.save(state, best_path)


def check_checkpoint_model_type(checkpoint_path):
    """
    检查checkpoint文件中使用的模型类型
    
    Args:
        checkpoint_path: checkpoint文件路径
    
    Returns:
        模型类型和配置
    """
    import torch
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 首先尝试从model_config中获取模型类型
        if 'model_config' in checkpoint and 'name' in checkpoint['model_config']:
            model_type = checkpoint['model_config']['name']
            return {
                'model_type': model_type,
                'model_config': checkpoint['model_config']
            }
        
        # 如果没有model_config，尝试从state_dict的键分析模型类型
        state_dict = checkpoint.get('model_state_dict', {})
        keys = list(state_dict.keys())
        
        if any('conv' in key for key in keys):
            return {
                'model_type': 'conv1d', 
                'model_config': {},
                'note': '模型类型通过权重键推断，可能不准确'
            }
        else:
            return {
                'model_type': 'mlp',
                'model_config': {},
                'note': '模型类型通过权重键推断，可能不准确'
            }
    
    except Exception as e:
        return {
            'error': f"无法加载checkpoint: {str(e)}",
            'model_type': 'unknown'
        } 