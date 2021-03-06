# based on examples from https://github.com/pytorch/examples
# https://github.com/L1aoXingyu/pytorch-beginner/https://github.com/L1aoXingyu/pytorch-beginner/
# https://github.com/bfarzin/pytorch_aae/blob/master/main_aae.py
# https://github.com/artemsavkin/aae/blob/master/aae.ipynb
import argparse
import os
import torch
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.nn import functional as F
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

import numpy as np

def save_reconstructs(model, z, epoch):
        with torch.no_grad():
            sample = model(z).cpu()
            save_image(sample.view(z.shape[0], 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

def save_random_reconstructs(model, nz, epoch):
        with torch.no_grad():
            sample = torch.randn(64, nz).to(device)
            sample = model(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

class Encoder(nn.Module):
    """Encodes high dimensional data point into
    a low dimension latent space. It is also considered as
    the generator in this AAE model so it is trained to fool the discriminator.
    """

    def __init__(self, nin, nout, nh1, nh2):
        super(Encoder, self).__init__()
        self.nin = nin
        self.nout = nout
        self.main = nn.Sequential(
            nn.Linear(nin, nh1),
            nn.ReLU(),
            nn.Linear(nh1, nh2),
            nn.ReLU(),
            nn.Linear(nh2, nout),
        )

    def forward(self, input):
        return self.main(input.view(-1, self.nin))


class Decoder(nn.Module):
    """Reconstructs a input from the latent, low dimensional space, into
    the original data space.
    """

    def __init__(self, nin, nout, nh1, nh2):
        """ """
        super(Decoder, self).__init__()
        self.nin = nin
        self.nout = nout
        self.main = nn.Sequential(
            nn.Linear(nin, nh1),
            nn.ReLU(),
            nn.Linear(nh1, nh2),
            nn.ReLU(),
            nn.Linear(nh2, nout),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """A discriminator module for the AAE. It is trained to discriminate between
    a "true" random sample from the latent space (sampled by the choice
    distribution, e.g. gaussian), and a "false" vector generated by the encoder
    from an input data point.
    """

    def __init__(self, nin, nh1, nh2):
        """dimensions of the input layer, the 1st and 2nd hidden layers."""
        super(Discriminator, self).__init__()
        self.nin = nin
        self.main = nn.Sequential(
            nn.Linear(nin, nh1),
            nn.ReLU(),
            nn.Linear(nh1, nh2),
            nn.ReLU(),
            nn.Linear(nh2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


# parameters
xdim = 28*28
zdim = 20
h1dim = 28*28*50
h2dim = 400
batchSize = 128
epochs = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data", train=True, download=True, transform=transforms.ToTensor()
    ),
    batch_size=batchSize,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data", train=False, transform=transforms.ToTensor()),
    batch_size=batchSize,
    shuffle=True,
)


enc = Encoder(xdim, zdim, h1dim, h2dim).to(device)
dec = Decoder(zdim, xdim, h2dim, h1dim).to(device)
dis = Discriminator(zdim, h2dim, h2dim).to(device)

optim_dis = optim.Adam(dis.parameters())
optim_dec = optim.Adam(dec.parameters())
optim_enc = optim.Adam(enc.parameters())

enc_losses = []
dec_losses = []
dis_losses = []

bce = nn.BCELoss(reduction="mean")
mse = nn.MSELoss(reduction="mean")
l1 = nn.L1Loss(reduction="mean")

xs, l = iter(train_loader).next()
xs.shape
l.shape

zs = enc(xs.to(device))
zs.shape

ws = dis(zs)
ws.shape

ys = dec(zs)
ys.shape


for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        x = data.to(device)
        #x = data.view(-1,xdim).to(device)
        z_fake = enc(x)
        z_real = torch.randn(batch_size, zdim).to(device)

        # train discriminator
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        loss_dis_real = bce(dis(z_real), labels_real)
        loss_dis_fake = bce(dis(z_fake.detach()), labels_fake).to(device)
        dis.zero_grad() # what's the difference between optim.zer...
        # optim_dis.zero_grad()
        loss_dis = loss_dis_fake + loss_dis_real
        loss_dis.backward(retain_graph=True)
        optim_dis.step()
        dis_losses.append(loss_dis.item())

        # train encoder as generator
        labels_real = torch.ones(batch_size, 1).to(device)
        loss_enc_g = bce(dis(z_fake), labels_real)
        loss_enc = loss_enc_g
        enc.zero_grad()
        loss_enc.backward(retain_graph=True)
        optim_enc.step()
        enc_losses.append(loss_enc.item())

        # train enc+dec for reconstruction
        rec = dec(z_fake.detach())
        loss_dec_rec = l1(x.view(-1,xdim), rec)
        dec.zero_grad()
        loss_dec = loss_dec_rec
        loss_dec.backward()
        optim_dec.step()
        dec_losses.append(loss_dec.item())

        if not (idx % len(train_loader)):
            print("epoch", epoch)
            print("mean losses:",
                    "dis loss", torch.mean(torch.FloatTensor(dis_losses)),
                    "enc loss", torch.mean(torch.FloatTensor(enc_losses)),
                    "dec loss", torch.mean(torch.FloatTensor(dec_losses))
                    )



######################################################## Test
epochs = 10
enc = Encoder(xdim, zdim, h1dim, h2dim).to(device)
dec = Decoder(zdim, xdim, h2dim, h1dim).to(device)
dis = Discriminator(zdim, h2dim, h2dim).to(device)
optim_dis = optim.Adam(dis.parameters())
optim_dec = optim.Adam(dec.parameters())
optim_enc = optim.Adam(enc.parameters())
dis_losses = []
enc_losses = []
dec_losses = []
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        x = data.to(device)
        #x = data.view(-1,xdim).to(device)
        z_real = torch.randn(batch_size, zdim).to(device)
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)

        # train enc
        dec.requires_grad_(False)
        dec.eval()
        enc.train()
        enc.requires_grad_(True)
        optim_enc.zero_grad()
        z_fake = enc(x)
        loss_enc_g = bce(dis(z_fake), labels_real)
        loss_enc = loss_enc_g
        loss_enc.backward()
        optim_enc.step()
        enc_losses.append(loss_enc.item())


        # train discriminator
        dis.train()
        enc.eval()
        dec.eval()
        enc.requires_grad_(False)
        dec.requires_grad_(False)
        dis.requires_grad_(True)
        optim_dis.zero_grad()
        z_fake = enc(x)

        loss_dis_real = bce(dis(z_real), labels_real)
        loss_dis_fake = bce(dis(z_fake), labels_fake)
        loss_dis = loss_dis_fake + loss_dis_real
        loss_dis.backward()
        optim_dis.step()
        dis_losses.append(loss_dis.item())

        # train dec
        dis.eval()
        enc.eval()
        dec.train()
        enc.requires_grad_(False)
        dec.requires_grad_(True)
        dis.requires_grad_(False)
        optim_dec.zero_grad()

        z_fake = enc(x)
        rec = dec(z_fake)
        loss_dec_rec = l1(x.view(-1,xdim), rec)
        loss_dec = loss_dec_rec
        loss_dec.backward()
        optim_dec.step()
        dec_losses.append(loss_dec.item())


        if idx % 50 == 0:
            print(epoch, idx, "losses:",
                    "dis loss", np.mean(dis_losses),
                    "enc loss", np.mean(enc_losses),
                    "dec loss", np.mean(dec_losses)
                    )




        # train encoder as generator
        labels_real = torch.ones(batch_size, 1).to(device)
        loss_enc_g = bce(dis(z_fake), labels_real)
        loss_enc = loss_enc_g
        enc.zero_grad()
        loss_enc.backward(retain_graph=True)
        optim_enc.step()
        enc_losses.append(loss_enc.item())

        # train enc+dec for reconstruction
        loss_dec_rec = l1(x.view(-1,xdim), rec)
        dec.zero_grad()
        loss_dec = loss_dec_rec
        loss_dec.backward()
        optim_dec.step()
        dec_losses.append(loss_dec.item())


save_random_reconstructs(dec, zdim, 101)

zs = enc(xs.cuda())

save_reconstructs(dec, zs, 989) 


imgs = ys.detach().cpu()
imgs = imgs.view(-1, 28, 28)
imgs.shape
grid = make_grid(imgs, nrow=16)

plt.imshow(grid.permute(1,2,0)

batch_tensor = torch.randn(10, 3, 256, 256)   # (N, C, H, W)
batch_tensor.shape

grid_img = make_grid(batch_tensor)

grid_img = make_grid(batch_tensor, nrow=5)

grid_img.shape

plt.imshow(grid_img.permute(1, 2, 0))

