import torch
import torch.nn as nn
import torch.nn.init as init

class MPBDBlock(nn.Module):
    """
    Multi-Path Block with Dual branches
    
    Args:
        input_channel: Size of input channels
        output_channel: Size of output channels
    """
    def __init__(self, input_channel=1, output_channel=4):
        super(MPBDBlock, self).__init__()
        
        # First branch with smaller kernel sizes
        self.Block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
        )
        
        # Second branch with larger kernel sizes
        self.Block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=True,
            ),
            nn.ReLU(),
        )
        
        # Downsample path if input and output channels differ
        self.downsample = nn.Sequential()
        if input_channel != output_channel:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )
    
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
    """
    def __init__(self, input_channel=1, embedding_c=50, kernel_size=3, overlap=1, padding=1):
        super(Embedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=embedding_c,
                kernel_size=kernel_size,
                stride=kernel_size-overlap,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(),
        )
    
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
    """
    def __init__(self, 
                 num_classes=3, 
                 list_inplanes=[3, 6, 18], 
                 num_rnn_sequence=18, 
                 embedding_c=50, 
                 seq_len=64,
                 dropout_rate=0.3):
        super(MPBDNet, self).__init__()
        
        # Validate and store parameters
        assert len(list_inplanes) < 6, "Too many block layers"
        
        self.seq_len = seq_len
        self.num_rnn_sequence = num_rnn_sequence
        self.list_inplanes = list_inplanes.copy()
        self.list_inplanes.insert(0, 1)  # Add input channel as first element
        self.embedding_c = embedding_c
        
        # Create MPBD blocks
        self.MPBDBlock_list = []
        for i in range(len(self.list_inplanes)-1):
            self.MPBDBlock_list.append(
                nn.Sequential(
                    MPBDBlock(self.list_inplanes[i], self.list_inplanes[i+1]),
                    MPBDBlock(self.list_inplanes[i+1], self.list_inplanes[i+1]),
                    MPBDBlock(self.list_inplanes[i+1], self.list_inplanes[i+1]),
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
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=fc1_input_features,
                out_features=256,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=256,
                out_features=128,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        
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
        # Ensure input shape is correct
        if len(x.shape) == 2:
            # Input is [batch_size, seq_len], add channel dimension
            x = x.unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # Pad if necessary to make length divisible by 8
        if x.size(2) % 8 != 0:
            padding_size = 8 - (x.size(2) % 8)
            x = torch.cat([x, torch.zeros([x.size(0), x.size(1), padding_size]).to(x.device)], dim=2)
        
        # Process through MPBD blocks
        x = self.MPBDBlock_list(x)
        x = torch.relu(x)
        
        # Embedding
        x = self.embedding(x)
        
        # Bidirectional LSTM processing
        x = (self.rnn(x)[0] + torch.flip(self.rnn2(torch.flip(x, dims=[1]))[0], dims=[1])) / 2
        
        # Fully connected layers
        x = self.fc1(x.flatten(1))
        x = self.fc2(x)
        x = self.output(x)
        
        return x 