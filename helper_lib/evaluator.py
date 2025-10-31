from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

DeviceLike = Union[str, torch.device]


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: DeviceLike) -> Tuple[float, float]:
    """
    Evaluate `model` on `data_loader`.
    """
    device = torch.device(device)
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        _, predictions = logits.max(dim=1)
        total_correct += predictions.eq(labels).sum().item()
        total_samples += labels.size(0)

    if total_samples == 0:
        raise ValueError("data_loader must contain at least one batch.")

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy
