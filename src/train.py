import math
import os
import shutil
import sys

import torch
import numpy as np
import torch.nn.functional as F

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

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