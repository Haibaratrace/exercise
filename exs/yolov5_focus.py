import torch
import torch.nn as nn


class Focus(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=1):
        super(Focus, self).__init__()

        self.layer = nn.Conv2d(in_channels=inchannel * 4, out_channels=outchannel, kernel_size=kernel)
        # self.ps = nn.PixelUnshuffle(2)
        # 6.0版本
        # self.focus = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=6, stride=2, padding=2)

    def forward(self, x):
        out = self.layer(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1))
        # out = self.layer(self.ps(x))
        # out = self.focus(x)
        return out


if __name__ == '__main__':
    x = torch.randn((1, 2, 12, 12))
    out, out2 = Focus(2, 1)(x)
    print(out, out2)
