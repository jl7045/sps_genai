from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DeviceLike = Union[str, torch.device]


def train(
    model: nn.Module,
    train_loader: DataLoader,
    device: DeviceLike,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    grad_clip: Optional[float] = None,
    progress: bool = True,
) -> nn.Module:
    """
    Train `model` using batches from `train_loader`.
    """
    if hasattr(train_loader, "__len__") and len(train_loader) == 0:
        raise ValueError("train_loader must contain at least one batch.")

    device = torch.device(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optimizer or optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False) if progress else train_loader

        for images, labels in iterator:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, preds = logits.max(dim=1)
            correct += preds.eq(labels).sum().item()
            total += batch_size

            if progress:
                iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{(100.0 * correct / max(total, 1)):.2f}%",
                )

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = 100.0 * correct / max(total, 1)
        print(f"[Train][Epoch {epoch}] loss={epoch_loss:.4f} acc={epoch_acc:.2f}%")

        if scheduler is not None:
            scheduler.step()

    return model
