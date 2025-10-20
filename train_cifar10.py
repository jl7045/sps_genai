import os
import torch
from helper_lib.data_loader import get_loader
from helper_lib.model import SimpleCNN
from helper_lib.trainer import train
from helper_lib.evaluator import evaluate

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader = get_loader(train=True, batch_size=128)
    test_loader = get_loader(train=False, batch_size=256)

    model = SimpleCNN(num_classes=10)
    model = train(model, train_loader, device, epochs=10, lr=1e-3)

    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, "models/cifar10_cnn.pt")
    print("Saved to models/cifar10_cnn.pt")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.2f}%")

if __name__ == "__main__":
    main()
