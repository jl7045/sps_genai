import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .energy_model import EnergyAutoencoder


def get_cifar10_dataloader(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"[EnergyModel] Using device: {device}")

    train_loader = get_cifar10_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = EnergyAutoencoder().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 统一保存到 app/models
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "energy_autoencoder_cifar10.pth")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if (i + 1) % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                print(
                    f"[EnergyModel][Epoch {epoch+1}/{args.epochs}] "
                    f"Step {i+1}/{len(train_loader)} - Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0

        torch.save(model.state_dict(), ckpt_path)
        print(f"[EnergyModel] Saved checkpoint to {ckpt_path}")

    print("[EnergyModel] Training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--no_cuda", action="store_true")
    # 默认统一到 app/models
    default_out = Path(__file__).resolve().parent / "models"
    parser.add_argument("--out_dir", type=str, default=str(default_out))
    args = parser.parse_args()

    train(args)
