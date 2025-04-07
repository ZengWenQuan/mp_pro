import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate=0.2, batch_norm=False):
        super().__init__()
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # 潜在空间
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        if batch_norm:
            encoder_layers.append(nn.BatchNorm1d(latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if batch_norm:
                decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers) 