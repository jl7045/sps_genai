import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

from .diffusion_model import SimpleUNet, DDPM


def get_cifar10_dataloader(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"[Diffusion] Using device: {device}")

    train_loader = get_cifar10_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    unet = SimpleUNet(in_channels=3, base_channels=64, time_dim=256)
    diffusion = DDPM(unet, timesteps=args.timesteps)
    diffusion.to(device)

    optimizer = optim.Adam(diffusion.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "diffusion_cifar10.pth")

    for epoch in range(args.epochs):
        diffusion.train()
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            t = torch.randint(0, args.timesteps,
                              (images.size(0),), device=device).long()

            loss = diffusion.p_losses(images, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_every == 0:
                print(
                    f"[Diffusion][Epoch {epoch+1}/{args.epochs}] "
                    f"Step {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}"
                )

        torch.save(diffusion.state_dict(), ckpt_path)
        print(f"[Diffusion] Saved checkpoint to {ckpt_path}")

    print("[Diffusion] Training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1000)

    # ✔ 统一为 app/models
    default_out = Path(__file__).resolve().parent / "models"
    parser.add_argument("--out_dir", type=str, default=str(default_out))

    args = parser.parse_args()
    train(args)
