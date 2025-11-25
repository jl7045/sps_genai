import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        """
        t: [batch]，整数时间步 0...T-1
        return: [batch, dim]
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        time_term = self.time_mlp(t_emb).view(t_emb.size(0), -1, 1, 1)
        h = h + time_term
        h = self.act(h)
        h = self.conv2(h)
        return self.act(h + self.skip(x))


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.down1 = ResidualBlock(in_channels, base_channels, time_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.pool = nn.AvgPool2d(2)

        self.mid = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)

        self.up1 = ResidualBlock(base_channels * 2, base_channels, time_dim)
        self.up2 = ResidualBlock(base_channels, base_channels, time_dim)

        self.out_conv = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        h1 = self.down1(x, t_emb)          # 32x32
        h2 = self.pool(h1)                 # 16x16
        h2 = self.down2(h2, t_emb)         # 16x16

        h_mid = self.mid(h2, t_emb)

        h = F.interpolate(h_mid, scale_factor=2, mode="nearest")  # 32x32
        h = self.up1(h, t_emb)
        h = self.up2(h, t_emb)

        out = self.out_conv(h)
        return out  # 预测噪声 epsilon


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000,
                 beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # 线性 beta 调度
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # α̅_{t-1}，第一项用 1 填充，长度保持 T
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=alphas_cumprod.dtype),
             alphas_cumprod[:-1]],
            dim=0,
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # posterior variance 公式：β_t * (1 - α̅_{t-1}) / (1 - α̅_t)，长度 = T
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_var", posterior_var)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_ac * x_start + sqrt_om * noise

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        noise_pred = self.model(x_noisy, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        sqrt_om_ac_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        eps_theta = self.model(x, t)

        model_mean = sqrt_recip_alpha_t * (x - betas_t * eps_theta / sqrt_om_ac_t)

        # t = 0 时不再加噪声
        if (t == 0).all():
            return model_mean

        posterior_var_t = self.posterior_var[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, batch_size, device):
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        return x.clamp(-1.0, 1.0)
