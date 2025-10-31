import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from .gan_models import Generator, Discriminator

BATCH_SIZE = 128
Z_DIM = 100
EPOCHS = 5
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models"
GEN_PATH = os.path.join(MODEL_DIR, "gan_generator.pt")

<<<<<<< HEAD
=======

>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620
def get_dataloader():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
<<<<<<< HEAD
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
=======
    ds = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=tfm
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620
    dataloader = get_dataloader()
    G = Generator(noise_dim=Z_DIM).to(DEVICE)
    D = Discriminator().to(DEVICE)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for real_imgs, _ in loop:
            real_imgs = real_imgs.to(DEVICE)
<<<<<<< HEAD
            b = real_imgs.size(0)

            real_label = torch.ones((b, 1), device=DEVICE)
            fake_label = torch.zeros((b, 1), device=DEVICE)
=======
            batch_size = real_imgs.size(0)

            real_label = torch.ones((batch_size, 1), device=DEVICE)
            fake_label = torch.zeros((batch_size, 1), device=DEVICE)
>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620

            D_real = D(real_imgs)
            loss_D_real = criterion(D_real, real_label)

<<<<<<< HEAD
            noise = torch.randn(b, Z_DIM, device=DEVICE)
=======
            noise = torch.randn(batch_size, Z_DIM, device=DEVICE)
>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620
            fake_imgs = G(noise).detach()
            D_fake = D(fake_imgs)
            loss_D_fake = criterion(D_fake, fake_label)

            loss_D = loss_D_real + loss_D_fake
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

<<<<<<< HEAD
            noise = torch.randn(b, Z_DIM, device=DEVICE)
=======
            noise = torch.randn(batch_size, Z_DIM, device=DEVICE)
>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620
            gen_imgs = G(noise)
            D_pred = D(gen_imgs)
            loss_G = criterion(D_pred, real_label)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loop.set_postfix({
                "loss_D": float(loss_D.item()),
                "loss_G": float(loss_G.item())
            })

        torch.save(
            {
                "state_dict": G.state_dict(),
                "z_dim": Z_DIM
            },
            GEN_PATH
        )

<<<<<<< HEAD
    print("done")

if __name__ == "__main__":
    train()
=======
    print("Training complete")


if __name__ == "__main__":
    train()


>>>>>>> 5eba60f6627c0b39a9a28073b2ea2c69feffe620
