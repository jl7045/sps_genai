import torch
import torch.nn as nn


class EnergyAutoencoder(nn.Module):
    """
    把自编码器当作 Energy Model:
    E(x) = ||x - f(x)||^2
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()

        # Encoder: 32x32 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),   # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),           # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),          # 8 -> 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)

        # Decoder: 4x4 -> 32x32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, in_channels, 4, 2, 1),  # 16 -> 32
            nn.Sigmoid(),  # 输出 [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_mu(h)
        return z

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), 256, 4, 4)
        x_rec = self.decoder(h)
        return x_rec

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec

    def energy(self, x):
        """
        E(x) = MSE(x, f(x)), 按样本取均值，返回 [batch]
        """
        x_rec = self.forward(x)
        mse = (x - x_rec) ** 2
        return mse.view(mse.size(0), -1).mean(dim=1)


def get_energy_model(device: str = "cpu", ckpt_path: str | None = None):
    model = EnergyAutoencoder()
    model.to(device)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"[EnergyModel] Loaded checkpoint from {ckpt_path}")
    model.eval()
    return model

