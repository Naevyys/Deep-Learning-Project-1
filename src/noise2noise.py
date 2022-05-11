import torch
from torch import nn
from torch.nn import functional as F
# from src.utils import utils
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

    def __init__(self, params):
        super().__init__()

        self.img_ch = params["img_channel"]
        self.h = params["height"]
        self.w = params["width"]

        self.enc_conv0 = nn.Conv2d(self.img_ch, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv2 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1,), padding='same')
        self.enc_conv4 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1,), padding='same')
        self.enc_conv6 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv5a = nn.Conv2d(96, 96,kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv5b = nn.Conv2d(96, 96,kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv4a = nn.Conv2d(144, 96,kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv4b = nn.Conv2d(96, 96,kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv3a = nn.Conv2d(144, 96,kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv3b = nn.Conv2d(96, 96,kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv2a = nn.Conv2d(144, 96,kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv2b = nn.Conv2d(96, 96,kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1a = nn.Conv2d(96+self.img_ch, 64,kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1b = nn.Conv2d(64, 32,kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1 = nn.Conv2d(32, self.img_ch,kernel_size=3, stride=(1, 1), padding='same')

        self.lre = torch.nn.LeakyReLU(0.1)
        self.mp = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.upscale2d = nn.Upsample(scale_factor=2, mode='nearest')

    # def upscale2d(self, x):
    #     factor = 2
    #     s = x.shape
    #     x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    #     x = torch.tile(x, [1, 1, 1, factor, 1, factor])
    #     x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    #     return x

    def forward(self, x):

        skips = [x]

        x = self.lre(self.enc_conv0(x))
        x = self.lre(self.enc_conv1(x))
        x = self.mp(x)
        skips.append(x)

        x = self.lre(self.enc_conv2(x))
        x = self.mp(x)
        skips.append(x)

        x = self.lre(self.enc_conv3(x))
        x = self.mp(x)
        skips.append(x)

        x = self.lre(self.enc_conv4(x))
        x = self.mp(x)
        skips.append(x)

        x = self.lre(self.enc_conv5(x))
        x = self.mp(x)
        x = self.lre(self.enc_conv6(x))

        x = self.upscale2d(x)

        x = torch.concat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv5a(x))
        x = self.lre(self.dec_conv5b(x))

        x = self.upscale2d(x)
        x = torch.concat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv4a(x))
        x = self.lre(self.dec_conv4b(x))

        x = self.upscale2d(x)
        x = torch.concat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv3a(x))
        x = self.lre(self.dec_conv3b(x))

        x = self.upscale2d(x)
        x = torch.concat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv2a(x))
        x = self.lre(self.dec_conv2b(x))

        x = self.upscale2d(x)

        x = torch.concat([x, skips.pop()], dim=1)

        x = self.lre(self.dec_conv1a(x))

        x = self.lre(self.dec_conv1b(x))

        x = self.dec_conv1(x)

        return x
