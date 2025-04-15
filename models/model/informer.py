import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from math import sqrt


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self.mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self.mask = indicator.view(scores.shape).to(device)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 计算采样的Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # 使用均匀采样而不是随机采样，更加稳定
        # 确保sample_k至少为1，避免除零错误
        sample_k = max(1, sample_k)
        if L_K < sample_k:
            sample_k = L_K
        step = L_K // sample_k
        index_sample = torch.arange(0, L_K, step).long() # 均匀采样
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # 找到稀疏性最高的Top_k查询
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        if n_top > L_Q:
            n_top = L_Q
        M_top = M.topk(n_top, sorted=False)[1]

        # 使用缩减的Q计算Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # 使用掩码（自注意力）
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)  # 应用dropout到注意力权重

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        
        return context_in, None

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # 计算样本数和top-k数
        U_part = self.factor * np.ceil(np.log(max(L_K, 1.1))).astype('int').item()  # 确保log参数大于1
        u = self.factor * np.ceil(np.log(max(L_Q, 1.1))).astype('int').item()  # 确保log参数大于1

        # 确保U_part和u至少为1
        U_part = max(1, U_part)
        u = max(1, u)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # 计算稀疏注意力分数
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # 添加缩放因子
        scale = self.scale or 1. / sqrt(D)
        scores_top = scores_top * scale
        
        # 获取初始上下文
        context = self._get_initial_context(values, L_Q)
        # 使用选定的top-k查询更新上下文
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # 自注意力部分
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # 前馈网络部分
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Informer(nn.Module):
    """
    简化版Informer模型，适配当前项目的输入输出格式
    """
    def __init__(self, input_dim, output_dim, d_model=256, n_heads=8, e_layers=3,
                 d_ff=1024, dropout=0.1, activation='gelu', output_attention=False):
        super(Informer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_attention = output_attention
        
        # 输入映射层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 使用Informer的概率注意力机制
        attention = AttentionLayer(
            ProbAttention(False, 5, attention_dropout=dropout, output_attention=output_attention),
            d_model, 
            n_heads
        )
        
        # 编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention,
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, output_dim)
        )
    
    def forward(self, x):
        """
        x: 输入张量 [batch_size, features]
        """
        # 特征降维
        x = self.input_projection(x)
        
        # 添加序列维度（对于单点预测）
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 通过编码器
        enc_out, attns = self.encoder(x)
        
        # 取出序列维度
        enc_out = enc_out.squeeze(1)  # [batch_size, d_model]
        
        # 通过输出层
        outputs = self.output_layer(enc_out)
        
        if self.output_attention:
            return outputs, attns
        else:
            return outputs 