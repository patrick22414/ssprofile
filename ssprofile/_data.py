from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import _utils

CIFAR10_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR10_STD = (24703233, 0.24348505, 0.26158768)

TRAIN_SET = CIFAR10(
    root=_utils.get_awnas_dataset_dir("cifar10"),
    train=True,
    transform=transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    ),
)

TEST_SET = CIFAR10(
    root=_utils.get_awnas_dataset_dir("cifar10"),
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    ),
)

DATALOADERS = {
    "train": DataLoader(TRAIN_SET, batch_size=64, shuffle=True, num_workers=4),
    "test": DataLoader(TEST_SET, batch_size=64, shuffle=False, num_workers=4),
}
