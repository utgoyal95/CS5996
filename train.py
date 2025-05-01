# train.py
import os
import argparse
import pathlib
import torch
import torch.optim as optim
import torchvision.utils as vutils
from utils import get_dataloaders, Generator, Discriminator, weights_init

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",   type=str,   default="MNIST")
parser.add_argument("--data_root", type=str,   default="./dcgan_outputs/data")
parser.add_argument("--img_size",  type=int,   default=64)
parser.add_argument("--batch",     type=int,   default=128)
# ... other hyperparams ...
args = parser.parse_args()

train_loader, test_loader, nc = get_dataloaders(
    args.data_root,
    args.dataset,
    args.img_size,
    args.batch,
)


# Config
DATA_DIR = pathlib.Path("./data")
RESULTS_DIR = pathlib.Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 64
BATCH    = 128
EPOCHS   = 25
NZ       = 100
LR       = 2e-4
BETA1    = 0.5

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def train():
    # Data loaders
    train_loader, _ = get_mnist_loaders(DATA_DIR, IMG_SIZE, BATCH)

    # Models
    # then pass `nc` into your model constructors:
    netG = Generator(nz=NZ, ngf=64, nc=nc).to(DEVICE)
    netD = Discriminator(nc=nc, ndf=64).to(DEVICE)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss and optimizers
    criterion = torch.nn.BCELoss()
    optimD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

    fixed_noise = torch.randn(64, NZ, 1, 1, device=DEVICE)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        for i, (real, _) in enumerate(train_loader):
            real = real.to(DEVICE)
            b = real.size(0)
            # Train D
            netD.zero_grad()
            label_real = torch.ones(b, device=DEVICE)
            out_real = netD(real).view(-1)
            lossD_real = criterion(out_real, label_real)

            noise = torch.randn(b, NZ, 1, 1, device=DEVICE)
            fake = netG(noise)
            label_fake = torch.zeros(b, device=DEVICE)
            out_fake = netD(fake.detach()).view(-1)
            lossD_fake = criterion(out_fake, label_fake)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimD.step()

            # Train G
            netG.zero_grad()
            out_gen = netD(fake).view(-1)
            lossG = criterion(out_gen, label_real)
            lossG.backward()
            optimG.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] Step {i}/{len(train_loader)}  LossD: {lossD.item():.4f}  LossG: {lossG.item():.4f}")

        # save sample grid
        with torch.no_grad():
            samples = netG(fixed_noise).cpu()
        vutils.save_image(samples, RESULTS_DIR / f"fake_epoch_{epoch:03d}.png", normalize=True, nrow=8)

    # Save weights
    torch.save(netG.state_dict(), RESULTS_DIR / "netG_final.pth")
    torch.save(netD.state_dict(), RESULTS_DIR / "netD_final.pth")


if __name__ == "__main__":
    print(f"Training on device: {DEVICE}")
    train()