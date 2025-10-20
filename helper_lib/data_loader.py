# helper_lib/data_loader.py
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2023, 0.1994, 0.2010)

def build_transforms(train: bool):
    aug = [transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose([
        transforms.Resize((64, 64)),
        *aug,
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

def get_loader(data_dir: str = "./data", batch_size: int = 128, train: bool = True, num_workers: int = 2):
    ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=build_transforms(train))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers)

def class_names():
    return ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
