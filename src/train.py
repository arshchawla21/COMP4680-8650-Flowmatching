import math
import os
import shutil
import sys

import torch
import numpy as np
import torch.nn.functional as F

def flow_matching_loss(model, x, pred_type, loss_type):
    B = x.shape[0]
    t = torch.rand(size=B, device=x.device).clamp(1e-3, 1-1e-3)
    eps = torch.randn_like(x)
    z_t = (1 - t[:, None]) * x + t[:, None] * eps

    pred = model(z_t, t)

    if pred_type == 'x' and loss_type == 'x':
        target, output = x, pred
    elif pred_type == 'v' and loss_type == 'v':
        target, output = eps - x, pred

    return F.mse_loss(target, output)

def train_one_epoch(model, dataloader, optim, device, epoch, pred_type, loss_type):
    model.train(True)
    optim.zero_grad()

    for x in dataloader:
        x = x.to(device, non_blocking=True)        # (B, D)
        loss = flow_matching_loss(model, x, pred_type, loss_type)
        
        if epoch % 50 == 0:
            print(f"Loss: {loss}")

        optim.zero_grad()
        loss.backward()
        optim.step()