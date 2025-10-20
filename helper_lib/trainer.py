import torch
from torch import nn, optim
from tqdm import tqdm

def train(
    model,
    train_loader,
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0*correct/total:.2f}%")

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"[Train] loss={epoch_loss:.4f} acc={epoch_acc:.2f}%")

    return model
