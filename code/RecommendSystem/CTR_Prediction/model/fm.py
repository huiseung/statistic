import torch
import torch.nn as nn
from layer import *


class FactorizationMachineModel(nn.Module):
    def __init__(self, field_dims, embedding_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.interaction = FactorizationMachineModel(field_dims, embedding_dim)

    def forward(self, x):
        return self.linear(x)+self.interaction(x)