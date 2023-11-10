import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
        self.layer = nn.Upsample(2)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode='nearest')
        # out = self.layer(x)
        return out


class Downsample(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(0.1))

    def forward(self, x):
        out = self.layer(x)
        return out
