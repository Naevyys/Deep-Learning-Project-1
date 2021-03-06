import torch
from torch import nn


class Noise2noise(nn.Module):
    """
    The Noise2noise model as described in the reference paper.
    """

    def __init__(self, params):
        super().__init__()

        self.img_ch = params["img_channel"]

        self.enc_conv0 = nn.Conv2d(self.img_ch, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv2 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1,), padding='same')
        self.enc_conv4 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1,), padding='same')
        self.enc_conv6 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv5a = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv5b = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv4a = nn.Conv2d(144, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv4b = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv3a = nn.Conv2d(144, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv3b = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv2a = nn.Conv2d(144, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv2b = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1a = nn.Conv2d(96 + self.img_ch, 64, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1b = nn.Conv2d(64, 32, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1 = nn.Conv2d(32, self.img_ch, kernel_size=3, stride=(1, 1), padding='same')

        self.lre = torch.nn.LeakyReLU(0.1)
        self.hsig = torch.nn.Hardsigmoid()
        self.mp = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.upscale2d = nn.Upsample(scale_factor=2, mode='nearest')

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
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv5a(x))
        x = self.lre(self.dec_conv5b(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv4a(x))
        x = self.lre(self.dec_conv4b(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv3a(x))
        x = self.lre(self.dec_conv3b(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv2a(x))
        x = self.lre(self.dec_conv2b(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv1a(x))
        x = self.lre(self.dec_conv1b(x))

        x = self.hsig(self.dec_conv1(x))

        return x


class Noise2noiseSimplified1(nn.Module):
    """
    First simplified version of the Noise2noise model: all skip connections are preserved, only the convolutions,
    maxpool and upsample layers between the last skip and the first concatenation are removed. Here images get
    downscaled to 2x2 and not 1x1.
    """

    def __init__(self, params):
        super().__init__()

        self.img_ch = params["img_channel"]

        self.enc_conv0 = nn.Conv2d(self.img_ch, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv2 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1,), padding='same')
        self.enc_conv4 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')

        self.dec_conv5a = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv5b = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv4a = nn.Conv2d(144, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv4b = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv3a = nn.Conv2d(144, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv3b = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv2a = nn.Conv2d(144, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv2b = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1a = nn.Conv2d(96 + self.img_ch, 64, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1b = nn.Conv2d(64, 32, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1 = nn.Conv2d(32, self.img_ch, kernel_size=3, stride=(1, 1), padding='same')

        self.lre = torch.nn.LeakyReLU(0.1)
        self.hsig = torch.nn.Hardsigmoid()
        self.mp = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.upscale2d = nn.Upsample(scale_factor=2, mode='nearest')

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

        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv5a(x))
        x = self.lre(self.dec_conv5b(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv4a(x))
        x = self.lre(self.dec_conv4b(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv3a(x))
        x = self.lre(self.dec_conv3b(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv2a(x))
        x = self.lre(self.dec_conv2b(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv1a(x))
        x = self.lre(self.dec_conv1b(x))

        x = self.hsig(self.dec_conv1(x))

        return x


class Noise2noiseSimplified2(nn.Module):
    """
    Second simplified version of the Noise2noise model: all the layers between the third skip and the second
    concatenation are removed. Here images get downscaled to 4x4 at minimum.
    """

    def __init__(self, params):
        super().__init__()

        self.img_ch = params["img_channel"]

        self.enc_conv0 = nn.Conv2d(self.img_ch, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv2 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1,), padding='same')

        self.dec_conv4 = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv3a = nn.Conv2d(144, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv3b = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv2a = nn.Conv2d(144, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv2b = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1a = nn.Conv2d(96 + self.img_ch, 64, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1b = nn.Conv2d(64, 32, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1 = nn.Conv2d(32, self.img_ch, kernel_size=3, stride=(1, 1), padding='same')

        self.lre = torch.nn.LeakyReLU(0.1)
        self.hsig = torch.nn.Hardsigmoid()
        self.mp = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.upscale2d = nn.Upsample(scale_factor=2, mode='nearest')

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

        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv4(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv3a(x))
        x = self.lre(self.dec_conv3b(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv2a(x))
        x = self.lre(self.dec_conv2b(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv1a(x))
        x = self.lre(self.dec_conv1b(x))

        x = self.hsig(self.dec_conv1(x))

        return x


class Noise2noiseSimplified3(nn.Module):
    """
    Third simplified version of the Noise2noise model: all the layers between the second skip and the third
    concatenation are removed. Here images get downscaled to 8x8 at minimum.
    """

    def __init__(self, params):
        super().__init__()

        self.img_ch = params["img_channel"]

        self.enc_conv0 = nn.Conv2d(self.img_ch, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv2 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')

        self.dec_conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv2a = nn.Conv2d(144, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv2b = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1a = nn.Conv2d(96 + self.img_ch, 64, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1b = nn.Conv2d(64, 32, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1 = nn.Conv2d(32, self.img_ch, kernel_size=3, stride=(1, 1), padding='same')

        self.lre = torch.nn.LeakyReLU(0.1)
        self.hsig = torch.nn.Hardsigmoid()
        self.mp = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.upscale2d = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        skips = [x]

        x = self.lre(self.enc_conv0(x))
        x = self.lre(self.enc_conv1(x))
        x = self.mp(x)
        skips.append(x)

        x = self.lre(self.enc_conv2(x))
        x = self.mp(x)
        skips.append(x)

        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv3(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv2a(x))
        x = self.lre(self.dec_conv2b(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv1a(x))
        x = self.lre(self.dec_conv1b(x))

        x = self.hsig(self.dec_conv1(x))

        return x


class Noise2noiseSimplified4(nn.Module):
    """
    Last simplified version of the Noise2noise model: all the layers between the first skip and the last
    concatenation are removed. Here images get downscaled to 16x16 at minimum.
    """

    def __init__(self, params):
        super().__init__()

        self.img_ch = params["img_channel"]

        self.enc_conv0 = nn.Conv2d(self.img_ch, 48, kernel_size=3, stride=(1, 1), padding='same')
        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=3, stride=(1, 1), padding='same')

        self.dec_conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1a = nn.Conv2d(96 + self.img_ch, 64, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1b = nn.Conv2d(64, 32, kernel_size=3, stride=(1, 1), padding='same')
        self.dec_conv1 = nn.Conv2d(32, self.img_ch, kernel_size=3, stride=(1, 1), padding='same')

        self.lre = torch.nn.LeakyReLU(0.1)
        self.hsig = torch.nn.Hardsigmoid()
        self.mp = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.upscale2d = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        skips = [x]

        x = self.lre(self.enc_conv0(x))
        x = self.lre(self.enc_conv1(x))
        x = self.mp(x)
        skips.append(x)

        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv2(x))

        x = self.upscale2d(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.lre(self.dec_conv1a(x))
        x = self.lre(self.dec_conv1b(x))

        x = self.hsig(self.dec_conv1(x))

        return x
