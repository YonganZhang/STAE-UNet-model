
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftThresholdAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.scale = dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.tau = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, N, C = x.shape
        Q = self.q_proj(x).view(B, N, self.n_heads, C // self.n_heads)
        K = self.k_proj(x).view(B, N, self.n_heads, C // self.n_heads)
        V = self.v_proj(x).view(B, N, self.n_heads, C // self.n_heads)

        attn = torch.einsum("bnhd,bmhd->bhnm", Q, K) * self.scale
        attn = F.relu(torch.where(torch.abs(attn) > self.tau, attn, torch.zeros_like(attn)))
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bmhd->bnhd", attn, V)
        out = out.reshape(B, N, C)
        return self.o_proj(out)
