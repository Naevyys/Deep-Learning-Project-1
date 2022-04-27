import torch
from torch import nn
from torch.nn import functional as F
#from src.utils import utils
import json

class TestNet(nn.Module):
    """
    Dummy network, without batchnorm, maxpool etc.
    Just a test with two convolutions and one upsample. 
    """
    def __init__(self, params):
        super().__init__()
    
        self.img_ch = params["img_channel"]
        self.h = params["height"]
        self.w = params["width"]

        self.conv1 = nn.Conv2d(self.img_ch, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, self.img_ch, kernel_size=5)
        self.up = torch.nn.Upsample(size=[self.h, self.w], mode='nearest')
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.up(x)
        x = self.sig(x)
        return x

class Noise2noise(nn.Module):
    """
    The noise to noise model to implement.  
    """
    def _init_(self, params):
        super()._init_()
        self.enc_conv0 = nn.Conv2d(3, 48, kernel_size=3, stride=[1,1,1,1],padding='same')
        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=3, stride=[1, 1, 1, 1], padding='same')

        self.lr = torch.nn.LeakyReLU(0.1)
        self.mp = torch.nn.MaxPool1d()

        raise NotImplementedError

    def forward(self, x):
        
        raise NotImplementedError

        return x