import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(data_dir="./data", batch_size=32, train=True):
    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2)
    return loader
