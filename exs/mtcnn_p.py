import torch
import torch.nn as nn


class P_Net(nn.Module):
    def __init__(self):
        super(P_Net, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3)),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))

        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3)),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU())

        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())

        self.conv_4_1 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.conv_4_2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(1, 1))
        self.conv_4_3 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        out1 = torch.softmax(self.conv_4_1(x), 1)
        out2 = self.conv_4_2(x)
        out3 = self.conv_4_3(x)
        return out1, out2, out3
