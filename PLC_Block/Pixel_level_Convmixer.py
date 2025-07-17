import torch
import torch.nn as nn
from einops import reduce


class CLA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        x_1 = reduce(x, "b c h w -> b c", "mean")
        x = self.dw(x)
        x_1_ = reduce(x, "b c h w -> b c", "mean")
        raise_ch = self.prob(x_1_ - x_1)
        att_score = torch.sigmoid(x_1_ + x_1_ * raise_ch)
        return torch.einsum("bchw, bc -> bchw", x, att_score)


class SLA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.prob = nn.Softmax2d()

    def forward(self, x):
        x_2 = reduce(x, "b c w h -> b w h", "mean")
        xp = self.pw(x)
        x_2_ = reduce(xp, "b c w h -> b w h", "mean")
        raise_ch = self.prob(x_2_ - x_2)
        att_score = torch.sigmoid(x_2_ + x_2_ * raise_ch)
        return torch.einsum("bchw, bc -> bchw", x, att_score)


class ConvMixer(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.CLA = nn.Sequential(
            CLA(in_channels),
            nn.GELU(),
            nn.BatchNorm2d(in_channels),
        )
        self.SLA = nn.Sequential(
            SLA(in_channels),
            nn.GELU(),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        x = x + self.CLA(x)
        x = self.SLA(x)
        return x
