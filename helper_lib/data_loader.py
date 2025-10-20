from __future__ import annotations

import os
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

Transform = Callable[[object], object]

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)


def get_transforms(
    train: bool = True,
    *,
    resize: Tuple[int, int] = (64, 64),
    normalize: bool = True,
    extra_transforms: Optional[Iterable[Transform]] = None,
) -> transforms.Compose:
    """
    Build the preprocessing pipeline used for CIFAR10 data.
    """
    pipeline: list[Transform] = [transforms.Resize(resize)]
    if train:
        pipeline.append(transforms.RandomHorizontalFlip())
    pipeline.append(transforms.ToTensor())
    if normalize:
        pipeline.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    if extra_transforms:
        pipeline.extend(extra_transforms)
    return transforms.Compose(pipeline)


def get_loader(
    data_dir: str = "./data",
    batch_size: int = 64,
    *,
    train: bool = True,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    transform: Optional[transforms.Compose] = None,
) -> DataLoader:
    """
    Create a DataLoader for the CIFAR10 dataset.
    """
    if num_workers is None:
        num_workers = min(8, max(1, (os.cpu_count() or 2) - 1))

    if shuffle is None:
        shuffle = train

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform or get_transforms(train=train),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def cifar10_class_names() -> Tuple[str, ...]:
    """Return class labels in the canonical CIFAR10 order."""
    return (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
