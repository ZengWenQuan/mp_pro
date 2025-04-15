import torch
import torch.nn as nn
import torch.nn.init as init

class MPBDBlock(nn.Module):
    """
    Multi-Path Block with Dual branches
    
    Args:
        input_channel: Size of input channels
        output_channel: Size of output channels
        batch_norm: Whether to use batch normalization
    """
    def __init__(self, input_channel=1, output_channel=4, batch_norm=False):
        super(MPBDBlock, self).__init__()
        
        # First branch with smaller kernel sizes
        block1_layers = [
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        ]
        if batch_norm:
            block1_layers.append(nn.BatchNorm1d(output_channel))
        block1_layers.append(nn.ReLU())
        
        block1_layers.append(
            nn.Conv1d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        if batch_norm:
            block1_layers.append(nn.BatchNorm1d(output_channel))
        block1_layers.append(nn.ReLU())
        
        self.Block1 = nn.Sequential(*block1_layers)
        
        # Second branch with larger kernel sizes
        block2_layers = [
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
            ),
        ]
        if batch_norm:
            block2_layers.append(nn.BatchNorm1d(output_channel))
        block2_layers.append(nn.ReLU())
        
        block2_layers.append(
            nn.Conv1d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=True,
            )
        )
        if batch_norm:
            block2_layers.append(nn.BatchNorm1d(output_channel))
        block2_layers.append(nn.ReLU())
        
        self.Block2 = nn.Sequential(*block2_layers)
        
        # Downsample path if input and output channels differ
        self.downsample = nn.Sequential()
        if input_channel != output_channel:
            downsample_layers = [
                nn.Conv1d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            ]
            if batch_norm:
                downsample_layers.append(nn.BatchNorm1d(output_channel))
            self.downsample = nn.Sequential(*downsample_layers)
    
    def forward(self, x):
        # Combine outputs from both branches and downsample path
        x = self.Block1(x) + self.Block2(x) + self.downsample(x)
        return x

class Embedding(nn.Module):
    """
    Embedding layer for sequence data
    
    Args:
        input_channel: Number of input channels
        embedding_c: Embedding dimension
        kernel_size: Kernel size for convolution
        overlap: Overlap between consecutive windows
        padding: Padding size
        batch_norm: Whether to use batch normalization
    """
    def __init__(self, input_channel=1, embedding_c=50, kernel_size=3, overlap=1, padding=1, batch_norm=False):
        super(Embedding, self).__init__()
        embedding_layers = [
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=embedding_c,
                kernel_size=kernel_size,
                stride=kernel_size-overlap,
                padding=padding,
                bias=True,
            ),
        ]
        if batch_norm:
            embedding_layers.append(nn.BatchNorm1d(embedding_c))
        embedding_layers.append(nn.ReLU())
        
        self.embedding = nn.Sequential(*embedding_layers)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Change from [B, C, L] to [B, L, C]
        return x

class MPBDNet(nn.Module):
    """
    Multi-Path Block with Dual branches Network
    
    Args:
        num_classes: Number of output classes/features
        list_inplanes: List of channel sizes for each block
        num_rnn_sequence: Number of RNN sequence steps
        embedding_c: Embedding dimension
        seq_len: Input sequence length
        dropout_rate: Dropout probability
        batch_norm: Whether to use batch normalization
    """
    def __init__(self, 
                 num_classes=3, 
                 list_inplanes=[3, 6, 18], 
                 num_rnn_sequence=18, 
                 embedding_c=50, 
                 seq_len=64,
                 dropout_rate=0.3,
                 batch_norm=False):
        super(MPBDNet, self).__init__()
        
        # Validate and store parameters
        assert len(list_inplanes) < 6, "Too many block layers"
        
        self.seq_len = seq_len
        self.num_rnn_sequence = num_rnn_sequence
        self.list_inplanes = list_inplanes.copy()
        self.list_inplanes.insert(0, 1)  # Add input channel as first element
        self.embedding_c = embedding_c
        self.batch_norm = batch_norm
        
        # Create MPBD blocks
        self.MPBDBlock_list = []
        for i in range(len(self.list_inplanes)-1):
            self.MPBDBlock_list.append(
                nn.Sequential(
                    MPBDBlock(self.list_inplanes[i], self.list_inplanes[i+1], batch_norm=batch_norm),
                    MPBDBlock(self.list_inplanes[i+1], self.list_inplanes[i+1], batch_norm=batch_norm),
                    MPBDBlock(self.list_inplanes[i+1], self.list_inplanes[i+1], batch_norm=batch_norm),
                    nn.AvgPool1d(3),
                )
            )
        self.MPBDBlock_list = nn.Sequential(*self.MPBDBlock_list)
        
        # Calculate sequence length after MPBD blocks
        # Each block has AvgPool1d with kernel_size=3, so divide by 3 for each block
        seq_after_blocks = seq_len
        for _ in range(len(self.list_inplanes)-1):
            seq_after_blocks = seq_after_blocks // 3
        
        # Embedding layer
        self.embedding = Embedding(
            input_channel=self.list_inplanes[-1],
            embedding_c=embedding_c,
            kernel_size=3,
            overlap=1,
            padding=1,
            batch_norm=batch_norm,
        )
        
        # Calculate sequence length after embedding
        # Embedding uses stride=2, so divide by 2
        seq_after_embedding = seq_after_blocks // 2
        
        # Bidirectional LSTM layers
        self.rnn = nn.LSTM(
            input_size=self.embedding_c,
            hidden_size=self.embedding_c,
            num_layers=1,
            batch_first=True,
        )
        self.rnn2 = nn.LSTM(
            input_size=self.embedding_c,
            hidden_size=self.embedding_c,
            num_layers=1,
            batch_first=True,
        )
        
        # Calculate input features for first fully connected layer
        fc1_input_features = seq_after_embedding * embedding_c
        
        # Fully connected layers
        fc1_layers = [
            nn.Linear(
                in_features=fc1_input_features,
                out_features=256,
            ),
        ]
        if batch_norm:
            fc1_layers.append(nn.BatchNorm1d(256))
        fc1_layers.append(nn.ReLU())
        fc1_layers.append(nn.Dropout(p=dropout_rate))
        self.fc1 = nn.Sequential(*fc1_layers)
        
        fc2_layers = [
            nn.Linear(
                in_features=256,
                out_features=128,
            ),
        ]
        if batch_norm:
            fc2_layers.append(nn.BatchNorm1d(128))
        fc2_layers.append(nn.ReLU())
        fc2_layers.append(nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(*fc2_layers)
        
        self.output = nn.Linear(
            in_features=128,
            out_features=num_classes,
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 检查批次大小，如果只有1个样本，暂时将BN层切换到eval模式
        batch_size = x.size(0)
        if batch_size == 1:
            # 保存所有BN层的当前状态
            bn_states = []
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    bn_states.append((m, m.training))
                    m.eval()  # 临时切换到评估模式
        
        # 确保输入形状正确
        if len(x.shape) == 2:
            # 输入是[batch_size, seq_len]，添加通道维度
            x = x.unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # 获取实际输入特征长度
        actual_seq_len = x.size(2)
        
        # 如果实际序列长度与模型初始化时的序列长度不一致，
        # 创建一个动态的全连接层来处理不同长度的输入
        is_dynamic_input = actual_seq_len != self.seq_len
        
        # 填充使得长度能被8整除（卷积网络需要）
        if x.size(2) % 8 != 0:
            padding_size = 8 - (x.size(2) % 8)
            x = torch.cat([x, torch.zeros([x.size(0), x.size(1), padding_size]).to(x.device)], dim=2)
        
        # 处理通过MPBD块
        x = self.MPBDBlock_list(x)
        x = torch.relu(x)
        
        # 嵌入
        x = self.embedding(x)
        
        # 双向LSTM处理
        x = (self.rnn(x)[0] + torch.flip(self.rnn2(torch.flip(x, dims=[1]))[0], dims=[1])) / 2
        
        # 扁平化并处理全连接层
        x_flat = x.flatten(1)  # [batch_size, seq_after_embedding*embedding_c]
        
        if is_dynamic_input:
            # 动态创建一个新的全连接层以处理当前输入尺寸
            dynamic_fc1 = nn.Linear(
                in_features=x_flat.size(1),
                out_features=256,
                device=x.device
            ).to(x.device)
            
            # 复制现有fc1的权重并进行适当调整
            # 只复制与新输入尺寸匹配的权重部分，或者填充不足的部分
            with torch.no_grad():
                # 获取现有fc1的第一个线性层
                original_fc1 = [m for m in self.fc1 if isinstance(m, nn.Linear)][0]
                
                # 复制权重和偏置（如果可能）
                if x_flat.size(1) <= original_fc1.in_features:
                    # 如果新输入维度更小，只复制需要的部分
                    dynamic_fc1.weight.copy_(original_fc1.weight[:, :x_flat.size(1)])
                    dynamic_fc1.bias.copy_(original_fc1.bias)
                else:
                    # 如果新输入维度更大，复制原有部分，其余初始化为零
                    dynamic_fc1.weight[:, :original_fc1.in_features].copy_(original_fc1.weight)
                    dynamic_fc1.bias.copy_(original_fc1.bias)
            
            # 应用动态全连接层
            x = dynamic_fc1(x_flat)
            
            # 应用其他层（激活函数、dropout等）
            for layer in [m for m in self.fc1 if not isinstance(m, nn.Linear)]:
                if batch_size == 1 and isinstance(layer, nn.BatchNorm1d):
                    # 对于单样本，跳过BatchNorm
                    continue
                x = layer(x)
        else:
            # 使用原始的fc1层
            if batch_size == 1:
                # 对于单样本，只应用线性层和非BN层
                fc1_linear = [m for m in self.fc1 if isinstance(m, nn.Linear)][0]
                x = fc1_linear(x_flat)
                for layer in [m for m in self.fc1 if not isinstance(m, nn.BatchNorm1d) and not isinstance(m, nn.Linear)]:
                    x = layer(x)
            else:
                x = self.fc1(x_flat)
        
        # 继续应用其他层
        if batch_size == 1:
            # 对于单样本，只应用线性层和非BN层
            fc2_linear = [m for m in self.fc2 if isinstance(m, nn.Linear)][0]
            x = fc2_linear(x)
            for layer in [m for m in self.fc2 if not isinstance(m, nn.BatchNorm1d) and not isinstance(m, nn.Linear)]:
                x = layer(x)
        else:
            x = self.fc2(x)
            
        x = self.output(x)
        
        # 恢复BN层的原始状态
        if batch_size == 1:
            for m, training in bn_states:
                m.training = training
        
        return x 