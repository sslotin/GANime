import numpy as np
import argparse
import itertools
import datetime
import time

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable # depricated, need to update and rewrite
import torch.autograd as autograd

from datasets import FacesDataset, AnimeDataset
from models import *

import torch
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0,
                    help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=4,
                    help='size of the batches')
parser.add_argument('--n_critic', type=int, default=5,
                    help='discriminator training steps per iter')
parser.add_argument('--sample_interval', type=int, default=5,
                    help='interval betwen image samples')
opt = parser.parse_args()


cycle_loss = torch.nn.L1Loss()

lambda_adv = 1
lambda_cycle = 10
lambda_gp = 10

G_A = Generator()
G_B = Generator()
D_A = Discriminator()
D_B = Discriminator()

if opt.epoch != 0:
    G_A.load_state_dict(torch.load('checkpoints/G_A_%d.pth' % opt.epoch))
    G_B.load_state_dict(torch.load('checkpoints/G_B_%d.pth' % opt.epoch))
    D_A.load_state_dict(torch.load('checkpoints/D_A_%d.pth' % opt.epoch))
    D_B.load_state_dict(torch.load('checkpoints/D_B_%d.pth' % opt.epoch))


dataloader_A = DataLoader(FacesDataset, batch_size=opt.batch_size, shuffle=True)
dataloader_B = DataLoader(FacesDataset, batch_size=opt.batch_size, shuffle=True)
optim_G = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()))
optim_D_A = torch.optim.Adam(D_A.parameters())
optim_D_B = torch.optim.Adam(D_B.parameters())


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = Variable(alpha * real_samples + (1 - alpha) * fake_samples, requires_grad=True)
    validity = D(interpolates)
    fake = Variable(torch.FloatTensor(np.ones(validity.data.shape)), requires_grad=True)
    gradients = autograd.grad(outputs=validity, inputs=interpolates, grad_outputs=fake,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def sample_images(batches_done): # could break?
    imgs = next(iter(dataloader_A))
    real_A = Variable(torch.FloatTensor(imgs))
    fake_B = G_B(real_A)
    AB = torch.cat((real_A.data, fake_B.data), -2)
    save_image(AB, 'images/%d.png' % batches_done, nrow=opt.batch_size, normalize=True)

print('Training started...')

batches_done = 0
prev_time = time.time()
for epoch in range(opt.n_epochs):
    for i, (batch_A, batch_B) in enumerate(zip(dataloader_A, dataloader_B)):
        print('...', i)
        # shouldn't we get rid of copy-pasting?
        imgs_A = Variable(torch.FloatTensor(batch_A))
        imgs_B = Variable(torch.FloatTensor(batch_B))

        optim_D_A.zero_grad()
        optim_D_B.zero_grad()

        fake_A = G_A(imgs_B).detach()
        fake_B = G_B(imgs_A).detach()

        gp_A = compute_gradient_penalty(D_A, imgs_A.data, fake_A.data)
        D_A_loss = - torch.mean(D_A(imgs_A)) + torch.mean(D_A(fake_A)) + lambda_gp * gp_A

        gp_B = compute_gradient_penalty(D_B, imgs_B.data, fake_B.data)
        D_B_loss = - torch.mean(D_B(imgs_B)) + torch.mean(D_B(fake_B)) + lambda_gp * gp_B

        D_loss = D_A_loss + D_B_loss
        D_loss.backward()

        optim_D_A.step()
        optim_D_B.step()

        if i % opt.n_critic == 0:
            optim_G.zero_grad()
            fake_A = G_A(imgs_B)
            fake_B = G_B(imgs_A)

            recov_A = G_A(fake_B)
            recov_B = G_B(fake_A)

            G_adv = - torch.mean(D_A(fake_A)) - torch.mean(D_B(fake_B))
            G_cycle = cycle_loss(recov_A, imgs_A) + cycle_loss(recov_B, imgs_B)
            G_loss = lambda_adv * G_adv + lambda_cycle * G_cycle

            G_loss.backward()
            optim_G.step()

        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)
            
            total = min(len(dataloader_A), len(dataloader_B))
            batches_left = opt.n_epochs * total  - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / opt.n_critic)
            prev_time = time.time()

            print("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, cycle: %f] ETA: %s"
                             % (epoch, opt.n_epochs, i, total, D_loss.data[0],
                                G_adv.data[0], G_cycle.data[0], time_left))

        batches_done += 1

        torch.save(G_B.state_dict(), 'checkpoints/G_B_%d.pth' % epoch)
        torch.save(G_A.state_dict(), 'checkpoints/G_A_%d.pth' % epoch)
        torch.save(D_A.state_dict(), 'checkpoints/D_A_%d.pth' % epoch)
        torch.save(D_B.state_dict(), 'checkpoints/D_B_%d.pth' % epoch)