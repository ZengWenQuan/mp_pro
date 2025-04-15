import torch
import torch.nn as nn
import torch.nn.init as init

class Conv1D(nn.Module):
    def __init__(self, input_channels=1, seq_len=64, num_classes=1, 
                 channels=[16, 32, 64, 128], kernel_sizes=[5, 5, 3, 3], 
                 fc_dims=[256, 128], dropout_rate=0.2, batch_norm=False):
        """
        Lightweight 1D Convolutional Neural Network for sequence prediction
        
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
        self.seq_len = seq_len
        
        # 确保卷积层和核大小列表长度一致
        assert len(channels) == len(kernel_sizes), "通道数列表和核大小列表长度必须相同"
        
        # 初始设置
        self.adaptive_pool = None
        curr_seq_len = seq_len
        
        # Convolutional layers with MaxPooling
        layers = []
        in_channels = input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            
            # 每个卷积层后都添加池化层以快速降维
            pool_size = 4 if i == 0 else 2  # 第一层使用更大的池化窗口
            layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            curr_seq_len = curr_seq_len // pool_size
            
            layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
        
        # 使用自适应池化确保输出维度固定，无论输入长度如何变化
        target_seq_len = 16  # 减小目标序列长度以减少参数
        self.adaptive_pool = nn.AdaptiveMaxPool1d(target_seq_len)
        curr_seq_len = target_seq_len
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算展平后的特征维度
        self.conv_output_size = channels[-1] * curr_seq_len
        
        # 全连接层
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
        
        # 计算并打印模型参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Conv1D模型参数量: {total_params:,}")
    
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
        
        # 处理卷积层
        x = self.conv_layers(x)
        
        # 应用自适应池化
        x = self.adaptive_pool(x)
        
        # 展平张量，准备送入全连接层
        x = x.view(x.size(0), -1)  # Flatten
        
        # 全连接层
        x = self.fc_layers(x)
        
        return x 