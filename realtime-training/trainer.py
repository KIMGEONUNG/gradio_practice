#!/usr/bin/env python

from __future__ import print_function

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import gradio as gr
from pycomar.samples import get_img


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):

  def __init__(self, ngf, nc, nz, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 64 x 64
    )

  def forward(self, input):
    return self.main(input)


class Discriminator(nn.Module):

  def __init__(self, ndf, nc, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
        # input is (nc) x 64 x 64
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 32
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 4 x 4
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid())

  def forward(self, input):
    return self.main(input)


class Trainer(object):

  def __init__(self):
    self.manualSeed = 999
    random.seed(self.manualSeed)
    torch.manual_seed(self.manualSeed)
    self.dataroot = "data/celeba"
    self.workers = 2
    self.batch_size = 128
    self.image_size = 64
    self.nc = 3
    self.nz = 100
    self.ngf = 64
    self.ndf = 64
    self.num_epochs = 5
    self.lr = 0.0002
    self.beta1 = 0.5
    self.ngpu = 1

    self.img_list = []
    self.G_losses = []
    self.D_losses = []
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    self.dataset = dset.ImageFolder(root=self.dataroot,
                                    transform=transforms.Compose([
                                        transforms.Resize(self.image_size),
                                        transforms.CenterCrop(self.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5)),
                                    ]))
    # Create the dataloader
    self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.workers)

    # Decide which device we want to run on
    self.device = torch.device("cuda:0" if (
        torch.cuda.is_available() and self.ngpu > 0) else "cpu")

    # Plot some training images
    self.real_batch = next(iter(self.dataloader))

    self.netG = Generator(ngf=self.ngf, nc=self.nc, nz=self.nz,
                          ngpu=self.ngpu).to(self.device)
    self.netG.apply(weights_init)

    self.netD = Discriminator(ndf=self.ndf, nc=self.nc,
                              ngpu=self.ngpu).to(self.device)
    self.netD.apply(weights_init)

    # Initialize BCELoss function
    self.criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

    # Establish convention for real and fake labels during training
    self.real_label = 1.
    self.fake_label = 0.

    # Setup Adam optimizers for both G and D
    self.optimizerD = optim.Adam(self.netD.parameters(),
                                 lr=self.lr,
                                 betas=(self.beta1, 0.999))
    self.optimizerG = optim.Adam(self.netG.parameters(),
                                 lr=self.lr,
                                 betas=(self.beta1, 0.999))

    # GUI
    self.cnt = 1
    self.on_training = True
    self.img_preview = get_img(1)

    with gr.Blocks() as self.demo:
      btn_start_train = gr.Button("Start Training")
      self.preview = gr.Image().style(height=300)
      with gr.Row():
        btn_stop_train = gr.Button("Stop Training")
        btn_resume_train = gr.Button("Resume Training")

      # EVENTS
      self.demo.load(self.update, inputs=None, outputs=self.preview, every=1)
      btn_start_train.click(self.train_start)
      btn_stop_train.click(self.off_toogle)
      btn_resume_train.click(self.on_toogle)

  def update(self):
    if len(self.img_list) == 0:
      return None
    img = transforms.ToPILImage()(self.img_list[-1])
    return img

  def on_toogle(self):
    print('on_toogle')
    self.on_training = True

  def off_toogle(self):
    print('off_toogle')
    self.on_training = False

  def launch(self):
    self.demo.queue().launch()

  def train_start(self):
    iters = 0

    for epoch in range(self.num_epochs):
      for i, data in enumerate(self.dataloader, 0):

        if not self.on_training:
          print("Stop Training")
          while not self.on_training:
            print('.',  end='')
          print("Resume Training")
          

        self.netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(self.device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size, ),
                           self.real_label,
                           dtype=torch.float,
                           device=self.device)
        # Forward pass real batch through D
        output = self.netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        # Generate fake image batch with G
        fake = self.netG(noise)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output = self.netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()

        self.netG.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizerG.step()

        # Output training stats
        if i % 50 == 0:
          print(
              '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, self.num_epochs, i, len(self.dataloader), errD.item(),
                 errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        self.G_losses.append(errG.item())
        self.D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 50 == 0) or ((epoch == self.num_epochs - 1) and
                                  (i == len(self.dataloader) - 1)):
          with torch.no_grad():
            fake = self.netG(self.fixed_noise).detach().cpu()
          self.img_list.append(
              vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


if __name__ == "__main__":
  gui = Trainer()
  gui.launch()
