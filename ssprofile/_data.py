from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import _utils

CIFAR10_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR10_STD = (24703233, 0.24348505, 0.26158768)

TRAIN_SET = CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    ),
)

VAL_SET = CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    ),
)

DATALOADERS = {
    "train": DataLoader(TRAIN_SET, batch_size=256, shuffle=True, num_workers=4),
    "val": DataLoader(VAL_SET, batch_size=256, shuffle=False, num_workers=4),
}
