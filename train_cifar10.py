import torch, os
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Model: 简单 CNN ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),  # 32->16
            nn.Conv2d(16,32,3,stride=1,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),  # 16->8
            nn.Flatten(),
            nn.Linear(32*8*8, 100), nn.ReLU(),
            nn.Linear(100, num_classes)
        )
    def forward(self, x): return self.net(x)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_train)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=2)

    model = SimpleCNN().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        total, correct, running = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            running += loss.item()
            pred = out.argmax(1)
            total += y.size(0)
            correct += (pred==y).sum().item()
        print(f"Epoch {epoch+1}: loss={running/len(train_loader):.4f}, acc={correct/total:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, "models/cifar10_cnn.pt")
    print("Saved => models/cifar10_cnn.pt")

if __name__ == "__main__":
    main()
