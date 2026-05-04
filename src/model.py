import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def sinusoidal_embd(t, dim=128):
    # t: (B,) tensor in [0, 1]
    k = dim // 2
    i = torch.arange(k, device=t.device, dtype=t.dtype)
    omega = torch.exp(-i * math.log(10000) / (k - 1))     # (k,)
    arg = t[:, None] * omega[None, :]                     # (B, k)
    return torch.cat([arg.sin(), arg.cos()], dim=-1)      # (B, dim)

class JiM(nn.Module):
    """Just image MLP :)"""
    def __init__(
        self,
        hidden_dim: int = 256,
        D: int = 2,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(D+128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, D)
        )

    def forward(self, z, t):
        et = sinusoidal_embd(t)
        g = torch.cat([z, et], dim=-1) # (D+128)
        return self.layers(g)