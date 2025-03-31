import torch
import torch.nn as nn

class Conv1D(nn.Module):
    def __init__(self, input_channels=1, seq_len=64, num_classes=1, 
                 channels=[32, 64, 128], kernel_sizes=[3, 3, 3], 
                 fc_dims=[256, 128], dropout_rate=0.2):
        """
        Simple 1D Convolutional Neural Network for sequence prediction
        
        Args:
            input_channels: Number of input channels
            seq_len: Sequence length
            num_classes: Number of output classes
            channels: List of channel sizes for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            fc_dims: List of fully connected layer dimensions
            dropout_rate: Dropout probability
        """
        super(Conv1D, self).__init__()
        
        # 保存输入参数
        self.input_channels = input_channels
        
        # Convolutional layers
        conv_layers = []
        in_channels = input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(2))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate size after convolutions and pooling
        # 确保序列长度是2的幂次方的最小整数
        # 这样可以避免序列长度过小导致MaxPool1d后长度为0的问题
        min_seq_len = 2 ** len(channels)  # 最小需要的序列长度
        if seq_len < min_seq_len:
            seq_len = min_seq_len
            print(f"Warning: seq_len too small, adjusted to {seq_len}")
        
        # 计算卷积和池化后的输出大小
        output_size = seq_len
        for _ in range(len(channels)):
            output_size = output_size // 2  # MaxPool1d with kernel_size=2
        
        # Fully connected layers
        fc_layers = []
        fc_in_dim = channels[-1] * output_size
        
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(fc_in_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            fc_in_dim = fc_dim
        
        fc_layers.append(nn.Linear(fc_in_dim, num_classes))
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        """Forward pass"""
        # 确保输入的形状正确：[batch_size, input_channels, seq_len]
        # 当前输入x的形状可能是 [batch_size, seq_len]，需要增加通道维度
        if len(x.shape) == 2:
            # 添加通道维度
            x = x.unsqueeze(1)
        
        # 如果输入是[batch_size, feature_dim, seq_len]但feature_dim != input_channels
        # 我们需要调整通道维度
        elif x.shape[1] != 1 and self.conv_layers[0].in_channels == 1:
            # 将feature_dim视为seq_len的一部分，重新调整维度
            batch_size = x.shape[0]
            x = x.view(batch_size, 1, -1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x 