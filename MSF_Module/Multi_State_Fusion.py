import torch
import torch.nn as nn


class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw = nn.Conv2d(in_channels, out_channels, 1)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pw(x)
        x = self.down(x)
        return x


class Fussion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 4, 4, 0)
        self.conv2 = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = nn.Conv2d(128, 128, 1, 1, 0)
        self.conv4 = nn.Conv2d(128, 128, 1, 1, 0)

        self.down1 = down(64, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 128)

        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.w3 = nn.Parameter(torch.tensor(1.0))
        self.w4 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x1, x2, x3, x4):
        d1 = self.down1(x1)
        o1 = self.conv1(d1)
        d2 = self.down2(x2 + d1)
        o2 = self.conv2(d2)
        d3 = self.down3(x3 + d2)
        o3 = self.conv3(d3)
        o4 = self.conv4(x4 + d3)
        out = self.w1 * o1 + self.w2 * o2 + self.w3 * o3 + self.w4 * o4
        out = out.view(out.size(0), -1)
        return out
