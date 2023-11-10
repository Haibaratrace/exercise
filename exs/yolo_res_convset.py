import torch.nn as nn


class Conv_Block(nn.Module):
    def __init__(self, inchannel, outchannel, kernel, stride=1, padding=0):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=kernel, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class Res_Block(nn.Module):
    def __init__(self, inchannel):
        super(Res_Block, self).__init__()
        self.res = nn.Sequential(
            Conv_Block(inchannel, inchannel // 2, 1),
            Conv_Block(inchannel // 2, inchannel, 3, 1, 1)
        )

    def forward(self, x):
        out = x + self.res(x)
        return out


class Conv_Set(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Conv_Set, self).__init__()

        self.conv = nn.Sequential(
            Conv_Block(inchannel, outchannel, 1),
            Conv_Block(outchannel, inchannel, 3, 1, 1),
            Conv_Block(inchannel, outchannel, 1),
            Conv_Block(outchannel, inchannel, 3, 1, 1),
            Conv_Block(inchannel, outchannel, 1)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
