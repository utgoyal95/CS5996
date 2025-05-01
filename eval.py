# eval.py
import datetime
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torchvision.utils as vutils
from utils import Generator, Discriminator, get_mnist_loaders

# Config (match train.py)
DATA_DIR    = pathlib.Path("./dcgan_outputs/data")
RESULTS_DIR = pathlib.Path("./dcgan_outputs/results")
IMG_SIZE    = 64
BATCH       = 256
NZ          = 100
LABEL_FRAC  = 0.10
DEVICE      = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


@torch.no_grad()
def extract_feats(netD, loader, pool):
    feats, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        h = x
        acts = []
        for layer in netD.main:
            h = layer(h)
            if isinstance(layer, nn.Conv2d) and h.shape[2] >= 4 and h.shape[3] >= 4:
                acts.append(pool(h).cpu())
        pooled = torch.cat([a.flatten(1) for a in acts], 1)
        feats.append(pooled)
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def linear_probe(netD, train_all):
    # prepare loaders
    n_lab = int(len(train_all) * LABEL_FRAC)
    idx = np.random.choice(len(train_all), n_lab, replace=False)
    lab_loader = DataLoader(Subset(train_all, idx), BATCH, shuffle=False)
    test_loader = DataLoader(get_mnist_loaders(DATA_DIR, IMG_SIZE, BATCH)[1], BATCH, shuffle=False)

    # extract
    POOL = nn.AdaptiveMaxPool2d((4, 4))
    netD.eval()
    X_tr, y_tr = extract_feats(netD, lab_loader, POOL)
    X_te, y_te = extract_feats(netD, test_loader, POOL)

    # scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # classify
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te)) * 100
    msg = f"{datetime.datetime.now()}: Linear-probe ({LABEL_FRAC:.2%} labels) = {acc:.2f}%\n"
    print(msg)
    with open(RESULTS_DIR / "eval.txt", "a") as f:
        f.write(msg)


def latent_interpolation(netG, steps=10):
    z0 = torch.randn(1, NZ, 1, 1, device=DEVICE)
    z1 = torch.randn(1, NZ, 1, 1, device=DEVICE)
    alphas = torch.linspace(0, 1, steps, device=DEVICE).view(-1, 1, 1, 1)
    z = z0 + alphas * (z1 - z0)
    with torch.no_grad():
        imgs = netG(z).cpu()
    vutils.save_image(imgs, RESULTS_DIR / "interpolation.png", normalize=True, nrow=steps)


def nearest_neighbour_check(netG, train_all, num_fake=8, real_subset=6000):
    fake = netG(torch.randn(num_fake, NZ, 1, 1, device=DEVICE)).cpu()
    sub_idx = np.random.choice(len(train_all), real_subset, replace=False)
    real_imgs = torch.stack([train_all[i][0] for i in sub_idx])
    f_vec = fake.view(num_fake, -1)
    r_vec = real_imgs.view(real_subset, -1)
    nn_real = []
    for v in f_vec:
        dists = ((r_vec - v).pow(2)).sum(1)
        idx = torch.argmin(dists).item()
        nn_real.append(real_imgs[idx])
    grid = torch.cat([fake, torch.stack(nn_real)], 0)
    vutils.save_image(grid, RESULTS_DIR / "nearest_neighbour.png", normalize=True, nrow=num_fake)


def main():
    # load models
    netG = Generator(nz=NZ).to(DEVICE)
    netD = Discriminator().to(DEVICE)
    netG.load_state_dict(torch.load(RESULTS_DIR / "netG_final.pth", map_location=DEVICE))
    netD.load_state_dict(torch.load(RESULTS_DIR / "netD_final.pth", map_location=DEVICE))

    # quantitative
    train_all = dset.MNIST(root=DATA_DIR, train=True, download=False, transform=T.Compose([
        T.Resize(IMG_SIZE), T.ToTensor(), T.Normalize((0.5,), (0.5,))]))
    linear_probe(netD, train_all)

    # qualitative
    latent_interpolation(netG)
    nearest_neighbour_check(netG, train_all)


if __name__ == "__main__":
    main()