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
    
class JiM_MF(nn.Module):
    """Just image MLP w/ MeanFlow :)"""
    def __init__(
        self,
        hidden_dim: int = 256,
        D: int = 2,
        t_embd: int = 128,
        h_embd: int = 128,
    ):
        super().__init__()
        self.t_embd = t_embd
        self.h_embd = h_embd

        in_dim = D + t_embd + h_embd

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
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

    def forward(self, z, t, h):
        et = sinusoidal_embd(t)
        eh = sinusoidal_embd(h)
        g = torch.cat([z, et, eh], dim=-1)
        return self.layers(g)