import torch
import torch.nn as nn


class Center_Loss(nn.Module):
    def __init__(self, lamda, cls, n_features):
        super(Center_Loss, self).__init__()
        self.lamda = lamda
        self.center = nn.Parameter(torch.randn(cls, n_features), requires_grad=True)

    def forward(self, features, lables):
        batch_center = self.center.index_select(dim=0, index=lables.long())
        cls_count = torch.histc(lables, bins=int(max(lables).item() + 1), min=0, max=int(max(lables).item()))
        batch_cls = cls_count.index_select(dim=0, index=lables.long())

        loss = self.lamda / 2 * torch.mean(torch.sum((features - batch_center) ** 2, dim=1) / batch_cls)

        return loss
