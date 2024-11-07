import torch
import torch.nn as nn
import numpy as np

class LayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_dtype = input.dtype
        ans = super().forward(input.type(torch.float32))
        ans = ans.type(orig_dtype)
        return ans


class Mlp(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(emb_dim, emb_dim*4)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(emb_dim*4, emb_dim)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class GatedMlp(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.linear_gate = nn.Linear(emb_dim, emb_dim*4)
        self.linear_1 = nn.Linear(emb_dim, emb_dim*4)
        self.linear_2 = nn.Linear(emb_dim*4, emb_dim)
        self.gelu = nn.GELU()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.linear_gate(x)
        x = self.gelu(x)
        y = self.linear_1(x)
        x = x * y
        x = self.linear_2(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
