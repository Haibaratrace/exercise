import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Layer(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(CNN_Layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.Dropout(0.3),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class Down_Sample(nn.Module):
    def __init__(self, inchannel):
        super(Down_Sample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class Up_Sample(nn.Module):
    def __init__(self, c):
        super(Up_Sample, self).__init__()
        self.conv = nn.Conv2d(c, c // 2, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2)
        up = self.conv(up)
        out = torch.cat((r, up), dim=1)
        return out


class UNet(nn.Module):
    def __init__(self, ):
        super(UNet, self).__init__()
        self.c1 = CNN_Layer(3, 64)
        self.d1 = Down_Sample(64)
        self.c2 = CNN_Layer(64, 128)
        self.d2 = Down_Sample(128)
        self.c3 = CNN_Layer(128, 256)
        self.d3 = Down_Sample(256)
        self.c4 = CNN_Layer(256, 512)
        self.d4 = Down_Sample(512)

        self.c5 = CNN_Layer(512, 1024)

        self.u1 = Up_Sample(1024)
        self.c6 = CNN_Layer(1024, 512)
        self.u2 = Up_Sample(512)
        self.c7 = CNN_Layer(512, 256)
        self.u3 = Up_Sample(256)
        self.c8 = CNN_Layer(256, 128)
        self.u4 = Up_Sample(128)
        self.c9 = CNN_Layer(128, 64)

        self.pre = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(self.d1(x1))
        x3 = self.c3(self.d2(x2))
        x4 = self.c4(self.d3(x3))
        x5 = self.c5(self.d4(x4))
        u_1 = self.c6(self.u1(x5, x4))
        u_2 = self.c7(self.u2(u_1, x3))
        u_3 = self.c8(self.u3(u_2, x2))
        u_4 = self.c9(self.u4(u_3, x1))

        out = self.pre(u_4)
        return out
