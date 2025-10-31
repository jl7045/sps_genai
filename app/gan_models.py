import torch
import torch.nn as nn

<<<<<<< HEAD
=======

>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620
class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.fc = nn.Linear(noise_dim, 128 * 7 * 7)
        self.net = nn.Sequential(
<<<<<<< HEAD
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
=======
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        img = self.net(x)
        return img

<<<<<<< HEAD
=======

>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
<<<<<<< HEAD
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
=======
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
<<<<<<< HEAD
=======

>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620
