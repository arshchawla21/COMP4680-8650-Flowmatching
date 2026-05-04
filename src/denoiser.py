import torch
from model import JiM

class Denoiser():
    def __init__(
        self,
        model,
        steps=50,
        D=2
    ):
        self.model = model
        self.steps = steps
        self.D = D

    @torch.no_grad()
    def generate(self, bsz, pred_type, device):
        z = torch.randn(bsz, self.D, device=device)
        timesteps = torch.linspace(1.0, 0.0, self.steps+1, device=device)

        # ode
        for i in range(self.steps):
            if pred_type == 'v':
                t = timesteps[i].expand(bsz)
                t_next = timesteps[i + 1].expand(bsz)
                v_pred = self.model(z, t)
                z = z + (t_next - t)[:, None] * v_pred
            elif pred_type == 'x':
                t = timesteps[i].expand(bsz)
                t_next = timesteps[i + 1].expand(bsz)
                x_pred = self.model(z, t)
                v_pred = (z - x_pred) / t[:, None]            # convert to velocity
                z = z + (t_next - t)[:, None] * v_pred

        return z

    @torch.no_grad()
    def _euler_step(self, z, t, t_next):
        v_pred = self.model(z, t)
        z_next = z + (t_next - t) * v_pred
        return z_next