import torch
import torch.nn as nn
import torch.nn.functional as F


class Arc_Loss(nn.Module):
    def __init__(self, feature, cls):
        super(Arc_Loss, self).__init__()
        self.W = nn.Parameter(torch.randn(feature, cls), requires_grad=True)

    def forward(self, x, m=0.5):
        x = F.normalize(x)
        w = F.normalize(self.W)
        s = torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(w ** 2))
        cosa = torch.matmul(x, w) / s
        angle = torch.acos(cosa)

        loss = torch.exp(s * torch.cos(angle + m)) / (
                torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(s * cosa) + torch.exp(
            s * torch.cos(angle + m)))
        return loss
