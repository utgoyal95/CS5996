# utils.py
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc,      4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1,     4, 1, 0, bias=False), nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_dataloaders(
    data_root: str,
    dataset: str,
    image_size: int,
    batch_size: int,
):
    """
    Returns (train_loader, test_loader, nc) for the requested dataset.
    dataset: one of ["MNIST", "CIFAR10", "CelebA", "ImageFolder"]
    """
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])

    if dataset.upper() == "MNIST":
        transform = T.Compose(transform.transforms + [T.Normalize((0.5,), (0.5,))])
        ds_train = MNIST(data_root, True, download=True, transform=transform)
        ds_test  = MNIST(data_root, False, download=True, transform=transform)
        nc = 1

    elif dataset.upper() == "CIFAR10":
        transform = T.Compose(transform.transforms + [T.Normalize((0.5,)*3, (0.5,)*3)])
        ds_train = CIFAR10(data_root, True, download=True, transform=transform)
        ds_test  = CIFAR10(data_root, False, download=True, transform=transform)
        nc = 3

    elif dataset.upper() == "CELEBA":
        transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5,)*3, (0.5,)*3),
        ])
        ds_train = CelebA(data_root, split="train", download=True, transform=transform)
        ds_test  = CelebA(data_root, split="test",  download=True, transform=transform)
        nc = 3

    elif dataset.upper() == "IMAGEFOLDER":
        # expects data_root/train/, data_root/val/ subfolders
        transform = T.Compose(transform.transforms + [T.Normalize((0.5,)*3, (0.5,)*3)])
        ds_train = ImageFolder(os.path.join(data_root, "train"), transform=transform)
        ds_test  = ImageFolder(os.path.join(data_root, "val"),   transform=transform)
        nc = 3

    else:
        raise ValueError(f"Unknown dataset {dataset}")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader, nc