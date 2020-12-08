import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from models.stylegan_layers import EqualLinear


class AttributeMapper(nn.Module):
    def __init__(self, style_dim, lr_mlp=0.01):
        super().__init__()

        layers = []

        for i in range(6):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.fc = nn.Sequential(*layers)

    def forward(self, latent):
        latent = self.fc(latent)

        return latent
