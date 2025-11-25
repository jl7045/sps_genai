from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.model import SimpleCNN
from helper_lib.data_loader import cifar10_class_names

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3


def get_dataloader():
    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=tfm,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return train_loader


def train():
    train_loader = get_dataloader()

    # 这里要调用函数，注意有括号 ()
    class_names = cifar10_class_names()
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    model = SimpleCNN(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch [{epoch}/{EPOCHS}]  "
              f"Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    # 保存模型到 app/models/cifar10_cnn.pt
    save_path = Path(__file__).resolve().parent / "models" / "cifar10_cnn.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    train()
