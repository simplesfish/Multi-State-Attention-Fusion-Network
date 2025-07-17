import torch
import torch.nn as nn
import torch.nn.functional as F

from LGH_Block import FeatureExtraction
from MSF_Module import Fussion
from PLC_Block import ConvMixer


class Gate(nn.Module):
    def __init__(self, in_channels, kernel):
        super().__init__()
        self.convmixer = ConvMixer(in_channels, kernel)

    def forward(self, x):
        x = self.convmixer(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.state1 = nn.Sequential(
            FeatureExtraction(1, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.state2 = nn.Sequential(
            FeatureExtraction(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.state3 = nn.Sequential(
            FeatureExtraction(64, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.state4 = nn.Sequential(
            FeatureExtraction(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.gate1 = Gate(64, 3)
        self.gate2 = Gate(64, 3)
        self.gate3 = Gate(128, 3)
        self.gate4 = Gate(128, 3)

        self.fusion = Fussion()

        self.proj = nn.Linear(2048, 128)

    def forward(self, x):
        x1 = self.state1(x)
        g1 = self.gate1(x1)
        x2 = self.state2(x1)
        g2 = self.gate2(x2)
        x3 = self.state3(x2)
        g3 = self.gate3(x3)
        x4 = self.state4(x3)
        g4 = self.gate4(x4)
        x = x4.view(x4.size(0), -1)
        f = self.fusion(g1, g2, g3, g4)
        x = x + f
        x = self.proj(x)

        return x


class MuSAFu_Net(nn.Module):
    def __init__(self, way=15):
        super(MuSAFu_Net, self).__init__()
        self.embedding_net = Net()
        self.way = way

    def forward(self, support_images, support_labels, query_images):
        support_emb = self.embedding_net(support_images)
        query_emb = self.embedding_net(query_images)
        support_emb = F.normalize(support_emb, p=2, dim=1)
        query_emb = F.normalize(query_emb, p=2, dim=1)
        similarities = torch.matmul(query_emb, support_emb.t())

        way_num = self.way
        Q = query_images.size(0)
        logits = torch.zeros(Q, way_num, device=support_images.device)
        index = support_labels.unsqueeze(0).expand(Q, -1)
        logits.scatter_add_(1, index, similarities)
        counts = torch.bincount(support_labels, minlength=way_num).float()
        logits = logits / counts.unsqueeze(0)
        return logits
