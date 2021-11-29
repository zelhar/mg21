# based on examples from https://github.com/pytorch/examples
# https://github.com/L1aoXingyu/pytorch-beginner/https://github.com/L1aoXingyu/pytorch-beginner/
# https://github.com/bfarzin/pytorch_aae/blob/master/main_aae.py
import argparse
import os
import random
import torch
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


def trainOneEpoch(epoch, *args):
    pass

def trainDiscriminator(model):
    pass

class Discriminator(nn.Module):
    """A discriminator module for the AAE. It is trained to discriminate between
    a "true" random sample from the latent space (sampled by the choice
    distribution, e.g. gaussian), and a "false" vector generated by the encoder
    from an input data point.
    """
    def __init__(self, nin, nh1, nh2):
        """dimensions of the input layer, the 1st and 2nd hidden layers.
        """
        super(Discriminator, self).__init__()
        self.nin = nin
        self.main = nn.Sequential(
                nn.Linear(nin, nh1),
                nn.ReLU(),
                nn.Linear(nh1,nh2),
                nn.ReLU(),
                nn.Linear(nh2, 1),
                nn.Sigmoid()
                )

    def forward(self, input):
        return self.main(input)


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
        """
        """
        super(Decoder, self).__init__()
        self.nin = nin
        self.nout = nout
        self.main = nn.Sequential(
                nn.Linear(nin, nh1),
                nn.ReLU(),
                nn.Linear(nh1, nh2),
                nn.ReLU(),
                nn.Linear(nh2, nout),
                nn.Sigmoid()
                )

    def forward(self, z):
        return self.main(z)


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

# check the dimensions of the data
xs, l = iter(train_loader).next()
xs.shape
l.shape

encoder = Encoder(xdim, zdim, h1dim, h2dim).to(device)
encoder.requires_grad_(True)
print(encoder)
decoder = Decoder(zdim, xdim, h2dim, h1dim).to(device)
print(decoder)
discriminator = Discriminator(zdim, h2dim, h1dim).to(device)
print(discriminator)
optimDisc = optim.Adam(discriminator.parameters())
optimEnc = optim.Adam(encoder.parameters())
optimEncGen = optim.Adam(encoder.parameters())
optimDec = optim.Adam(decoder.parameters())


for batch_idx, (xs, _) in enumerate(train_loader):
    # train discriminator on real and fake (=encoded) data
    optimEnc.zero_grad()
    optimDisc.zero_grad()
    optimDec.zero_grad()
    encoder.eval()
    decoder.eval()
    discriminator.train()
    encoder.requires_grad_(False)
    decoder.requires_grad_(False)
    discriminator.requires_grad_(True)
    currentBatchSize = xs.shape[0]
    real_labels = torch.ones(currentBatchSize, 1).to(device)
    fake_labels = torch.zeros(currentBatchSize, 1).to(device)

    xs = xs.to(device)
    zFake = encoder(xs)
    zRandom = torch.randn_like(zFake).to(device)
    criterion = nn.BCELoss()
    predictFake = discriminator(zFake.detach())
    predictReal = discriminator(zRandom)
    loss_real = criterion(predictReal, real_labels)
    loss_fake = criterion(predictFake, fake_labels)
    discrimLoss = loss_real + loss_fake
    discrimLoss.backward()
    optimDisc.step()

    # train the encoder as a generator, to fool the discriminator
    encoder.requires_grad_(True)
    encoder.train()
    discriminator.requires_grad_(False)
    discriminator.eval()
    decoder.requires_grad_(False)
    decoder.eval()
    zFake = encoder(xs)
    predictFake = discriminator(zFake)
    loss_gen = criterion(predictFake, real_labels)
    loss_gen.backward()
    optimEncGen.step()

    # train the encoder and decoder to recreate the original image
    discriminator.requires_grad_(False)
    encoder.requires_grad_(True)
    decoder.requires_grad_(True)
    discriminator.eval()
    encoder.train()
    decoder.train()
    z = encoder(xs)
    y = decoder(z)
    recon_loss = criterion(y, xs.view(-1, encoder.nin))
    recon_loss.backward()
    optimDec.step()
    optimEnc.step()
    if batch_idx % 50 == 0:
        print(discrimLoss.item(), loss_gen.item(), recon_loss.item())









# https://github.com/bfarzin/pytorch_aae/blob/master/main_aae.py









#Encoder
class Q_net(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss

# Decoder
class P_net(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)

# Discriminator
class D_net_gauss(nn.Module):  
    def __init__(self,N,z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))    

EPS = 1e-15
z_red_dims = 120
Q = Q_net(784,1000,z_red_dims).cuda()
P = P_net(784,1000,z_red_dims).cuda()
D_gauss = D_net_gauss(500,z_red_dims).cuda()

# Set learning rates
gen_lr = 0.0001
reg_lr = 0.00005

#encode/decode optimizers
optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
#regularizing optimizers
optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
optim_D = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)




def train(encoder, decoder, discriminator, optimEnc, optimDec, optimDisc, optimEncGen):
    for batch_idx, (xs, _) in enumerate(train_loader):
        # train discriminator on real and fake (=encoded) data
        optimEnc.zero_grad()
        optimEncGen.zero_grad()
        optimDisc.zero_grad()
        optimDec.zero_grad()
        encoder.eval()
        decoder.eval()
        discriminator.train()
        encoder.requires_grad_(False)
        decoder.requires_grad_(False)
        discriminator.requires_grad_(True)
        currentBatchSize = xs.shape[0]
        real_labels = torch.ones(currentBatchSize, 1).to(device)
        fake_labels = torch.zeros(currentBatchSize, 1).to(device)

        xs = xs.to(device)
        zFake = encoder(xs)
        zRandom = torch.randn_like(zFake).to(device)
        criterion = nn.BCELoss()
        predictFake = discriminator(zFake.detach())
        predictReal = discriminator(zRandom)
        loss_real = criterion(predictReal, real_labels)
        loss_fake = criterion(predictFake, fake_labels)
        discrimLoss = loss_real + loss_fake
        discrimLoss.backward()
        optimDisc.step()

        # train the encoder as a generator, to fool the discriminator
        encoder.requires_grad_(True)
        encoder.train()
        discriminator.requires_grad_(False)
        discriminator.eval()
        decoder.requires_grad_(False)
        decoder.eval()
        zFake = encoder(xs)
        predictFake = discriminator(zFake)
        loss_gen = criterion(predictFake, real_labels)
        loss_gen.backward()
        optimEncGen.step()

        # train the encoder and decoder to recreate the original image
        discriminator.requires_grad_(False)
        encoder.requires_grad_(True)
        decoder.requires_grad_(True)
        discriminator.eval()
        encoder.train()
        decoder.train()
        z = encoder(xs)
        y = decoder(z)
        recon_loss = criterion(y, xs.view(-1, encoder.nin))
        recon_loss.backward()
        optimDec.step()
        optimEnc.step()
        if batch_idx % 50 == 0:
            print(discrimLoss.item(), loss_gen.item(), recon_loss.item())



train(Q, P, D_gauss, optim_Q_enc, optim_P, optim_D, optim_Q_gen)

data_iter = iter(train_loader)
iter_per_epoch = len(train_loader)
total_step = 50000

# Start training
for step in range(total_step):

    # Reset the data_iter
    if (step+1) % iter_per_epoch == 0:
        data_iter = iter(train_loader)

    # Fetch the images and labels and convert them to variables
    images, labels = next(data_iter)
    images = images.to(device)

    #reconstruction loss
    P.zero_grad()
    Q.zero_grad()
    D_gauss.zero_grad()

    z_sample = Q(images)   #encode to z
    X_sample = P(z_sample) #decode to X reconstruction
    recon_loss = F.binary_cross_entropy(X_sample+EPS,images+EPS)

    recon_loss.backward()
    optim_P.step()
    optim_Q_enc.step()

    # Discriminator
    ## true prior is random normal (randn)
    ## this is constraining the Z-projection to be normal!
    Q.eval()
    #z_real_gauss = torch.randn(images.size()[0], z_red_dims) * 5..cuda()
    z_real_gauss = torch.randn_like(z_sample).to(device)
    D_real_gauss = D_gauss(z_real_gauss)

    z_fake_gauss = Q(images)
    D_fake_gauss = D_gauss(z_fake_gauss)

    D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))

    D_loss.backward()
    optim_D.step()

    # Generator
    Q.train()
    z_fake_gauss = Q(images)
    D_fake_gauss = D_gauss(z_fake_gauss)
    
    G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))

    G_loss.backward()
    optim_Q_gen.step()   

    
    if (step+1) % 100 == 0:
        # print ('Step [%d/%d], Loss: %.4f, Acc: %.2f' 
        #        %(step+1, total_step, loss.data[0], accuracy.data[0]))

        #============ TensorBoard logging ============#
        # (1) Log the scalar values
        info = {
            'recon_loss': recon_loss.data[0],
            'discriminator_loss': D_loss.data[0],
            'generator_loss': G_loss.data[0]
        }


