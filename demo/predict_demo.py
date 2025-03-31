import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# 设置matplotlib字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import MLP, Conv1D
from utils.general import load_config, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Demo prediction with trained model')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to model config (optional)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of random samples to generate')
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


def generate_random_samples(num_samples, seq_len=64):
    """Generate random samples for demonstration"""
    # Generate some sample data (sine waves with noise)
    x = np.linspace(0, 2*np.pi, seq_len)
    samples = []
    
    for i in range(num_samples):
        # Generate a sine wave with random frequency and phase
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(0.5, 2.0)
        
        # Add some noise
        noise_level = np.random.uniform(0.1, 0.3)
        noise = np.random.normal(0, noise_level, seq_len)
        
        # Create the sample
        sample = amplitude * np.sin(freq * x + phase) + noise
        samples.append(sample)
    
    return np.array(samples)


def predict(model, data, model_name):
    """Run prediction on input data"""
    device = get_device()
    
    # Convert data to tensor
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    
    # Prepare data based on model type
    if model_name == 'conv1d' and len(data.shape) == 2:
        # Add channel dimension for Conv1D
        data = data.unsqueeze(1)
    
    # Run prediction
    with torch.no_grad():
        data = data.to(device)
        predictions = model(data)
    
    return predictions.cpu().numpy()


def plot_samples_and_predictions(samples, predictions):
    """Plot input samples and their predictions"""
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Plot sample
        if num_samples > 1:
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]
        else:
            ax1 = axes[0]
            ax2 = axes[1]
        
        ax1.plot(samples[i])
        ax1.set_title(f"Sample {i+1}")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Value")
        ax1.grid(True)
        
        # Plot prediction
        pred_value = predictions[i][0]
        ax2.bar(["Prediction"], [pred_value], color='blue')
        ax2.set_title(f"Prediction: {pred_value:.4f}")
        ax2.set_ylabel("Value")
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Load model
    model, model_name = load_model(args.weights, config)
    
    # Generate random samples
    seq_len = 64  # Default sequence length
    if config and 'data' in config and 'seq_len' in config['data']:
        seq_len = config['data']['seq_len']
    
    samples = generate_random_samples(args.num_samples, seq_len)
    
    # Run prediction
    predictions = predict(model, samples, model_name)
    
    # Plot samples and predictions
    plot_samples_and_predictions(samples, predictions)
    
    print(f"Generated {args.num_samples} random samples and made predictions.")


if __name__ == '__main__':
    main() 