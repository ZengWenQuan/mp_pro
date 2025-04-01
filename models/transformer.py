import torch
import torch.nn as nn
import torch.nn.init as init
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        位置编码模块，为序列添加位置信息
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建一个足够长的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """前向传播"""
        # x: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return x

class SpectralTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout_rate=0.1, output_dim=3, batch_norm=False):
        """
        基于Transformer的光谱预测模型
        
        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            nhead: 多头注意力头数
            num_layers: Transformer编码器层数
            dim_feedforward: 前馈网络维度
            dropout_rate: Dropout概率
            output_dim: 输出维度（预测标签数量）
            batch_norm: 是否使用BatchNormalization
        """
        super(SpectralTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.batch_norm = batch_norm
        
        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward, output_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
        
        if batch_norm:
            self.output_bn = nn.BatchNorm1d(d_model)
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 打印输入形状，帮助调试
        orig_shape = x.shape
        
        # 确保输入形状正确 - Conv1D处理的是 [batch_size, channels, seq_len]
        # Transformer需要的是 [batch_size, seq_len, features]
        
        # 如果输入是2D: [batch_size, seq_len]，添加一个特征维度
        if len(x.shape) == 2:
            x = x.unsqueeze(2)  # 变为 [batch_size, seq_len, 1]
        
        # 如果输入是3D但是Conv1D格式 [batch_size, channels, seq_len]
        # 需要转置为Transformer格式 [batch_size, seq_len, channels]
        elif len(x.shape) == 3 and x.shape[1] != x.shape[2]:  # 形状不是方阵，判断可能是通道维度在中间
            x = x.transpose(1, 2)  # 交换轴，变为 [batch_size, seq_len, channels]
        
        # 检查特征维度是否与预期相符，不符则调整
        if x.shape[2] != self.input_dim:
            if self.input_dim == 1:
                # 如果期望的输入维度是1，则压缩最后一维
                batch_size, seq_len, _ = x.shape
                x = x.reshape(batch_size, seq_len, 1)
        
        # 嵌入层
        x = self.embedding(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)
        
        # 使用序列的平均值作为特征表示
        x = torch.mean(x, dim=1)
        
        # 输出层
        output = self.output_layer(x)
        
        if self.batch_norm:
            output = self.output_bn(output)
        
        return output 