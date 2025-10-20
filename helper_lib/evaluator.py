import torch
from torch import nn

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        total_correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * total_correct / total
    return avg_loss, acc
