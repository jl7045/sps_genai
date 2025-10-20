from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(train: bool = True):
    if train:
        tfm = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    return tfm

def get_loader(data_dir: str = "./data", batch_size: int = 64, train: bool = True, num_workers: int = 2):
    ds = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=get_transforms(train)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers)

def cifar10_class_names():
    return [
        "airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"
    ]
