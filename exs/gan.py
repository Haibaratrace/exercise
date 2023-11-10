import torch.nn as nn


class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 784),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer(x)
        return out
