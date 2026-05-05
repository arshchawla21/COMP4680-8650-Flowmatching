import torch
import numpy as np
import torch.nn.functional as F

def flow_matching_loss(model, x, pred_type, loss_type, P=None, opt=False):
    B = x.shape[0]
    D = x.shape[1]
    
    t = torch.rand(B, device=x.device).clamp(1e-3, 1-1e-3)

    if P is None or not opt:
        eps = torch.randn_like(x)
    else:
        # make eps is 2 dim (intrinsic) -> low rank noise
        # using "patching"
        P = torch.as_tensor(P, dtype=x.dtype, device=x.device)
        eps_2d = torch.randn(B, P.shape[0], device=x.device) 
        eps = eps_2d @ P                           # (B, D)

    z_t = (1 - t[:, None]) * x + t[:, None] * eps

    pred = model(z_t, t)

    if pred_type == 'x' and loss_type == 'x':
        target, output = x, pred
    elif pred_type == 'x' and loss_type == 'v':
        output, target = (z_t - pred) / t[:, None], eps - x
    elif pred_type == 'v' and loss_type == 'x':
        output, target = z_t - t[:, None] * pred, x
    elif pred_type == 'v' and loss_type == 'v':
        target, output = eps - x, pred

    return F.mse_loss(target, output)

def train_n_steps(model, dataloader, optim, device, n_steps, pred_type, loss_type, opt=False, log_every=500):
    model.train()
    losses = []
    step = 0
    data_iter = iter(dataloader)

    while step < n_steps:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)            # restart when exhausted
            x = next(data_iter)

        x = x.to(device, non_blocking=True)
        loss = flow_matching_loss(model, x, pred_type, loss_type, dataloader.dataset.P, opt=opt)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())
        step += 1
        if step % log_every == 0:
            recent = sum(losses[-log_every:]) / log_every
            print(f"step {step:>6}/{n_steps}  loss {recent:.6f}")

    return losses