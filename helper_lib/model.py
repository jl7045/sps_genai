import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """64x64 RGB -> Conv16 -> ReLU -> MaxPool -> Conv32 -> ReLU -> MaxPool -> Flatten -> FC100 -> ReLU -> FC10"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(32 * 16 * 16, 100)  # 64->32->16 after two pools
        self.fc2   = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 3x64x64 -> 16x64x64
        x = self.pool(x)            # -> 16x32x32
        x = F.relu(self.conv2(x))   # -> 32x32x32
        x = self.pool(x)            # -> 32x16x16
        x = torch.flatten(x, 1)     # -> 32*16*16
        x = F.relu(self.fc1(x))     # -> 100
        x = self.fc2(x)             # -> 10
        return x
