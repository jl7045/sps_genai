from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Convolutional neural network used for CIFAR10 classification.
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_size: Tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flattened_dim = self._infer_flatten_dim(input_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

        self.apply(self._init_weights)

    def _infer_flatten_dim(self, input_size: Tuple[int, int]) -> int:
        if len(input_size) != 2:
            raise ValueError("input_size must be a tuple of (height, width).")

        height, width = input_size
        dummy = torch.zeros(1, 3, height, width)
        with torch.no_grad():
            features = self.features(dummy)
        return int(torch.numel(features) / features.size(0))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)
