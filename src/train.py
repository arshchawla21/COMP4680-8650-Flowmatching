import torch
import numpy as np
import torch.nn.functional as F

def flow_matching_loss(model, x, pred_type, loss_type, clamp=0.01, P=None, opt=False):
    B = x.shape[0]
    t = torch.rand(B, device=x.device).clamp(clamp, 1-clamp)

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

def meanflow_loss(model, x, fm_ratio=0.5, clamp=0.01):
    B = x.shape[0]
    t = torch.rand(B, device=x.device).clamp(min=clamp)
    r = torch.rand(B, device=x.device) * t
    is_fm = torch.rand(B, device=x.device) < fm_ratio
    r = torch.where(is_fm, t, r)
    h = t - r

    eps = torch.randn_like(x)

    z_t = (1 - t[:, None]) * x + t[:, None] * eps
    v = eps - x

    # u expressed in terms of x-pred, JVP differentiates the whole thing
    def u_func(z_, t_, h_):
        x_pred = model(z_, t_, h_)
        return (z_ - x_pred) / t_[:, None]

    tgnt = (v, torch.ones_like(t), torch.ones_like(h))
    u, dudt = torch.func.jvp(u_func, (z_t, t, h), tgnt)

    u_tgt_mf = (v - h[:, None] * dudt).detach()
    u_tgt = torch.where(is_fm[:, None], v, u_tgt_mf)
    return F.mse_loss(u, u_tgt)

def train_n_steps(model, dataloader, optim, device, n_steps, pred_type, loss_type, clamp=0.01, opt=False, log_every=500, mf=False):
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
        loss = flow_matching_loss(model, x, pred_type, loss_type, clamp, dataloader.dataset.P, opt=opt, mf=mf)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())
        step += 1
        if step % log_every == 0:
            recent = sum(losses[-log_every:]) / log_every
            print(f"step {step:>6}/{n_steps}  loss {recent:.6f}")

    return losses

def train_n_steps_mf(model, dataloader, optim, device, n_steps, fm_ratio=0.5, clamp=0.01, log_every=500):
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
        loss = meanflow_loss(model, x, fm_ratio, clamp)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())
        step += 1
        if step % log_every == 0:
            recent = sum(losses[-log_every:]) / log_every
            print(f"step {step:>6}/{n_steps}  loss {recent:.6f}")

    return losses