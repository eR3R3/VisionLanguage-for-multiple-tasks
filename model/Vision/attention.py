import torch
from torch import nn
from vision_config import VisionConfig
from model.util import Mlp, LayerNorm
from model.util import LayerNorm

class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.attention_dropout = config.attention_dropout
        self.emb_dim = config.emb_dim
        self.mlp = Mlp(emb_dim=config.emb_dim)
        self.linear_qkv = nn.Linear(self.emb_dim, self.emb_dim * 3)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.layer_norm = LayerNorm(self.emb_dim)
        self.num_head = config.num_attention_head
        self.num_feature =config.num_feature
        self.head_dim = config.emb_dim // config.num_attention_head
        self.scale = self.head_dim ** -0.5
        self.batch_size = config.batch_size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_qkv(x)
        # [batch_size, num_feature, emb_dim*3]
        q, k, v = x.chunk(3, dim=-1)
        # 3 * [batch_size, num_feature, emb_dim]
        q = q.reshape(self.batch_size, self.num_feature, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(self.batch_size, self.num_feature, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(self.batch_size, self.num_feature, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        # 3 * [batch_size, num_head, num_feature, head_dim]
        attn_weights = (torch.matmul(q, k.transpose(2, 3)) * self.scale)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)
        x = torch.matmul(attn_weights, v)
        x = x.transpose(1, 2)
        x = x.reshape(self.batch_size, self.num_feature, self.emb_dim)
        return x

class ResidualTransformer(nn.Module):
    def __init__(self,config: VisionConfig, attention=VisionAttention, Mlp=Mlp):
        super().__init__()
        self.attention = attention(config)
        self.num_layer = config.num_layer
        self.mlp = Mlp(emb_dim=config.emb_dim)
        self.layer_norm = LayerNorm(config.emb_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layer):
            residual = x
            x = self.layer_norm(x)
            x = self.attention(x)
            x = self.mlp(x)
            x = x + residual
        return x








