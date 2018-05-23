import torch
import torch.nn as nn
import torch.nn.functional as F

# just regular layers from unet

class Down(nn.Module): # maybe should be implemented as functions too?
    def __init__(self, in_channels, out_channels, normalize=True, dropout=None):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers += [nn.InstanceNorm2d(out_channels, affine=True)]
        layers += [nn.LeakyReLU(0.2)]
        if dropout:
            layers += [nn.Dropout(dropout)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def Up(in_channels, out_channels, dropout=None):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
              nn.InstanceNorm2d(out_channels, affine=True),
              nn.ReLU(inplace=True)]
    if dropout:
        layers += [nn.Dropout(dropout)]
    return nn.Sequential(*layers)
    
def Generator(channels=3):
    return nn.Sequential(
        Down(channels, 64, normalize=False),
        Down(64, 128),
        Down(128, 256),
        Down(256, 256, dropout=0.5),
        Down(256, 256, dropout=0.5, normalize=False),
        Up(256, 256, dropout=0.5),
        Up(256, 256, dropout=0.5),
        Up(256, 128),
        Up(128, 64),
        nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid() # Tanh?
    )

def Discriminator(channels=3):
    def Block(in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers += [nn.BatchNorm2d(out_channels, 0.8)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        return layers

    return nn.Sequential(
        *Block(channels, 64, normalize=False),
        *Block(64, 128),
        *Block(128, 256),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(256, 1, kernel_size=4)
    )