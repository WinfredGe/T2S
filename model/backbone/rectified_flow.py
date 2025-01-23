import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
class RectifiedFlow:
    def euler(self, x_t, v, dt):
        x_t = x_t + v * dt
        return x_t
    def create_flow(self, x_1, t):
        x_0 = torch.randn_like(x_1).to(x_1.device)
        t = t[:, None, None]  # [B, 1, 1, 1]
        x_t = t * x_1 + (1 - t) * x_0
        return x_t, x_0
    def loss(self, v, noise_gt):
        # noise_gt : x_1 - x_0
        loss = F.mse_loss(v, noise_gt)
        return loss
if __name__ == '__main__':
    rf = RectifiedFlow()
    t =torch.tensor([0.999])
    x_t = rf.create_flow(torch.ones(1, 24, 1, ).float(), t)
    plt.plot(x_t[0].detach().cpu().numpy().squeeze())
    plt.plot(x_t[1].detach().cpu().numpy().squeeze())
    plt.show()

    print(x_t)
