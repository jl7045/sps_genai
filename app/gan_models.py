import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()

        # fc layer expands noise to 128 * 7 * 7
        self.fc = nn.Linear(noise_dim, 128 * 7 * 7)

        # deconvolutional layers
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 7x7 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1),    # 14x14 -> 28x28
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # matching the 28x28 single-channel output
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),     # 28 -> 14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),   # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.net(x)
