import torch
import torch.nn as nn
import torch.nn.init as init

class Conv1D(nn.Module):
    def __init__(self, input_channels=1, seq_len=64, num_classes=1, 
                 channels=[32, 64, 128], kernel_sizes=[3, 3, 3], 
                 fc_dims=[256, 128], dropout_rate=0.2, batch_norm=False):
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
            batch_norm: Whether to use batch normalization
        """
        super(Conv1D, self).__init__()
        
        # 保存输入参数
        self.input_channels = input_channels
        
        # 确保序列长度是2的幂次方的最小整数
        # 这样可以避免序列长度过小导致MaxPool1d后长度为0的问题
        min_seq_len = 2 ** len(channels)  # 最小需要的序列长度
        if seq_len < min_seq_len:
            seq_len = min_seq_len
            print(f"Warning: seq_len too small, adjusted to {seq_len}")
        
        # Convolutional layers
        layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算卷积层输出大小
        self.conv_output_size = channels[-1] * (seq_len // (2 ** len(channels)))
        
        # Fully connected layers
        fc_layers = []
        prev_dim = self.conv_output_size
        for dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            if batch_norm:
                fc_layers.append(nn.BatchNorm1d(dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        fc_layers.append(nn.Linear(prev_dim, num_classes))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # 使用Kaiming初始化所有卷积层和线性层
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重使用Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 对于卷积层使用Kaiming初始化
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # 对于批标准化层，权重初始化为1，偏置初始化为0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对于线性层使用Kaiming初始化
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # 确保输入的形状正确：[batch_size, input_channels, seq_len]
        if len(x.shape) == 2:
            # 输入为 [batch_size, seq_len]，需要增加通道维度
            x = x.unsqueeze(1)  # 变为 [batch_size, 1, seq_len]
        elif len(x.shape) == 3 and x.shape[1] != self.input_channels:
            # 输入为 [batch_size, feature_dim, seq_len] 但特征维度不匹配
            if self.input_channels == 1:
                # 如果期望单通道，则将所有特征合并为序列长度
                batch_size = x.shape[0]
                x = x.view(batch_size, 1, -1)
            else:
                # 如果期望多通道，但维度不匹配，输出错误信息
                raise ValueError(f"Conv1D: Expected input_channels={self.input_channels}, got {x.shape[1]}")
        
        # 执行卷积层操作
        try:
            x = self.conv_layers(x)
        except RuntimeError as e:
            print(f"Conv1D卷积层错误: {e}")
            print(f"输入形状: {x.shape}")
            raise
        
        # 展平张量，准备送入全连接层
        try:
            x = x.view(x.size(0), -1)  # Flatten
        except RuntimeError as e:
            print(f"Conv1D展平错误: {e}")
            print(f"卷积输出形状: {x.shape}")
            raise
        
        # 执行全连接层操作
        try:
            x = self.fc_layers(x)
        except RuntimeError as e:
            print(f"Conv1D全连接层错误: {e}")
            print(f"展平后形状: {x.shape}")
            raise
        
        return x 