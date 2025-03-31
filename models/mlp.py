import torch
import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[128, 256, 128], output_dim=1, dropout_rate=0.2):
        """
        Simple MLP model for prediction
        
        Args:
            input_dim: Input dimension size
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension size
            dropout_rate: Dropout probability
        """
        super(MLP, self).__init__()
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # 使用Kaiming初始化所有线性层
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重使用Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 对于ReLU激活函数，使用Kaiming正态分布初始化，增益为sqrt(2)
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # 打印输入形状，帮助调试
        orig_shape = x.shape
        
        # MLP期望输入形状为 [batch_size, feature_dim]
        if len(x.shape) == 3:
            # 输入为 [batch_size, seq_len, feature_dim] 或 [batch_size, feature_dim, seq_len]
            # 对于光谱数据，通常我们想要将序列或频道维度展平
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)  # 展平为 [batch_size, seq_len*feature_dim]
            print(f"MLP: Input flattened from {orig_shape} to {x.shape}")
        
        # 检查输入特征维度
        if x.shape[1] != self.model[0].in_features:
            print(f"MLP: 警告 - 输入特征维度 {x.shape[1]} 与模型期望的 {self.model[0].in_features} 不匹配")
            # 我们可以尝试适应这种情况，但这里只是输出警告
        
        try:
            return self.model(x)
        except RuntimeError as e:
            print(f"MLP前向传播错误: {e}")
            print(f"输入形状: {x.shape}")
            print(f"期望的输入特征: {self.model[0].in_features}")
            raise 