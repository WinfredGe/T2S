import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from typing import Optional
import torch.utils.data
def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)
class DDPM:
    def __init__(self,total_steps: int, device:torch.device):
        super().__init__()
        self.device = device
        self.beta = torch.linspace(0.0001, 0.02, total_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.total_steps = total_steps
        self.sigma2 = self.beta #sigma^2 = beta
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean.to(self.device), var.to(self.device)
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> [torch.Tensor, torch.Tensor]:
        if eps is None:
            eps = torch.randn_like(x0).to(self.device)
        mean, var = self.q_xt_x0(x0, t)
        return (mean + (var**0.5)*eps).to(self.device), eps
    def p_sample(self, xt: torch.Tensor,n_xt:torch.tensor, t: torch.Tensor) -> torch.Tensor:
        # n_xt = pred_noise_xt = eps_theta
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * n_xt)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps
    def loss(self, n_gt:torch.Tensor,n_xt: torch.Tensor,):
        return F.mse_loss(n_gt, n_xt)
if __name__ == '__main__':
    ddpm = DDPM(total_steps=100,device='cpu')
    x_0 = torch.ones(1, 6, 64)
    for t in range(1,100,10):
        t = torch.tensor([t]).to(ddpm.device)
        x_t = ddpm.q_sample(x_0, t)
        plt.imshow(x_t[0].detach().numpy(), cmap='gray')
        plt.show()
