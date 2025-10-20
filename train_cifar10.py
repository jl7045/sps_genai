from helper_lib.data_loader import get_loader
from helper_lib.model import CNNModel
from helper_lib.trainer import fit

def main():
    train_loader = get_loader("./data", batch_size=128, train=True)
    test_loader  = get_loader("./data", batch_size=256, train=False)
    model = CNNModel(num_classes=10)
    fit(
        model,
        train_loader,
        test_loader,
        epochs=5,
        lr=1e-3,
        out_path="./models/cifar10_cnn.pt"
    )

if __name__ == "__main__":
    main()
