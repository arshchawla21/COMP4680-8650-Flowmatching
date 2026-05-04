import torch
import torch.nn as nn
from model import JiM

class Denoiser(nn.Module):
    def __init__(
        self,
        data
    ):
        pass

    def sample_t(self, n: int, device=None):
        z = torch