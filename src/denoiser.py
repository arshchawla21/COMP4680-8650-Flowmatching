import torch

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
    def generate(self, bsz, pred_type, device, P=None, opt=False):
        if P is None or not opt:
            z = torch.randn(bsz, self.D, device=device)
        else:
            P = torch.as_tensor(P, device=device)
            z = torch.randn(bsz, P.shape[0], device=device) @ P

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
    
class Denoiser_MF():
    def __init__(
        self,
        model,
        steps=5,
        D=2
    ):
        self.model = model
        self.steps = steps
        self.D = D

    # @torch.no_grad()
    # def generate(self, bsz, device):
    #     z = torch.randn(bsz, self.D, device=device)

    #     interval = 1.0 / self.steps
    #     h = torch.full((bsz,), interval, device=device)
    #     for i in range(self.steps):
    #         t_val = 1.0 - i * interval                          # 1.0, 0.8, 0.6, ...
    #         t = torch.full((bsz,), t_val, device=device)
            
    #         u = self.model(z, t, h)
    #         z = z - interval * u
    #     return z
    @torch.no_grad()
    def generate(self, bsz, device,):
        z = torch.randn(bsz, self.D, device=device)

        interval = 1.0 / self.steps
        h = torch.full((bsz,), interval, device=device)
        for i in range(self.steps):
            t_val = 1.0 - i * interval
            t = torch.full((bsz,), t_val, device=device)
            x_pred = self.model(z, t, h)
            alpha = (h / t)[:, None]                  # = h/t, in [0, 1]
            z = (1 - alpha) * z + alpha * x_pred      # convex combo form
        return z

    @torch.no_grad()
    def _euler_step(self, z, t, t_next):
        v_pred = self.model(z, t)
        z_next = z + (t_next - t) * v_pred
        return z_next