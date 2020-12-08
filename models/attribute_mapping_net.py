import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function


class AttributeMapper(nn.Module):
    def __init__(self, style_dim):
        super().__init__()

        layers = []

        for i in range(6):
            layers.append(nn.Linear(style_dim, style_dim))
            layers.append(nn.ReLU(True))

        self.fc = nn.Sequential(*layers)

    def forward(self, latent):
        latent = self.fc(latent)

        return latent
