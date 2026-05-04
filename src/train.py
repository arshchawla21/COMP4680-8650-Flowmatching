import torch
import numpy as np
import torch.nn.functional as F

def flow_matching_loss(model, x, pred_type, loss_type):
    B = x.shape[0]
    t = torch.rand(B, device=x.device).clamp(1e-3, 1-1e-3)
    eps = torch.randn_like(x)
    z_t = (1 - t[:, None]) * x + t[:, None] * eps

    pred = model(z_t, t)

    if pred_type == 'x' and loss_type == 'x':
        target, output = x, pred
    elif pred_type == 'v' and loss_type == 'v':
        target, output = eps - x, pred

    return F.mse_loss(target, output)

def train_one_epoch(model, dataloader, optim, device, epoch, epoch_print, pred_type, loss_type):
    model.train(True)
    
    total_loss = 0.0
    num_batches = 0
    
    for x in dataloader:
        x = x.to(device, non_blocking=True)        # (B, D)
        loss = flow_matching_loss(model, x, pred_type, loss_type)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    if epoch % epoch_print == 0:
        print(f"=== Epoch: {epoch}, Avg Loss: {avg_loss:.6f} ===")
    
    return avg_loss