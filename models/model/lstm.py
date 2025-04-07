import torch
import torch.nn as nn
import torch.nn.init as init

class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, 
                 bidirectional=True, dropout_rate=0.2, output_dim=3, batch_norm=False):
        """
        LSTM模型用于光谱序列预测
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            bidirectional: 是否使用双向LSTM
            dropout_rate: Dropout概率
            output_dim: 输出维度（预测标签数量）
            batch_norm: 是否使用Batch Normalization
        """
        super(LSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.batch_norm = batch_norm
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim * self.num_directions)
        
        # 全连接层
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.constant_(param, 0)
    
    def forward(self, x):
        """前向传播"""
        # 打印输入形状，帮助调试
        orig_shape = x.shape
        
        # 确保输入形状正确 - Conv1D处理的是 [batch_size, channels, seq_len]
        # LSTM需要的是 [batch_size, seq_len, features]
        
        # 如果输入是2D: [batch_size, seq_len]，添加一个特征维度
        if len(x.shape) == 2:
            x = x.unsqueeze(2)  # 变为 [batch_size, seq_len, 1]
        
        # 如果输入是3D但是Conv1D格式 [batch_size, channels, seq_len]
        # 需要转置为LSTM格式 [batch_size, seq_len, channels]
        elif len(x.shape) == 3 and x.shape[1] != x.shape[2]:  # 形状不是方阵，判断可能是通道维度在中间
            x = x.transpose(1, 2)  # 交换轴，变为 [batch_size, seq_len, channels]
        
        # 检查特征维度是否与预期相符，不符则调整
        if x.shape[2] != self.input_dim:
            if self.input_dim == 1:
                # 如果期望的输入维度是1，则压缩最后一维
                batch_size, seq_len, _ = x.shape
                x = x.reshape(batch_size, seq_len, 1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        
        # 应用BatchNorm（如果启用）
        if self.batch_norm:
            last_hidden = self.bn(last_hidden)
        
        # 全连接层
        output = self.fc(last_hidden)
        
        return output 