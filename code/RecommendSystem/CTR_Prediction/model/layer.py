import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)
        self.offset = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.compat.long)
        self.init_weight()

    def forward(self, x):
        """

        :param x: Long Tensor, size: (batch_size, num_fields)
        :return: Float Tensor, size: (batch_size, num_fields, embedding_dim)
        """
        # index 맞추기 위해 더하는 행위가 backpropagation에 영향을 안 주기 위해 new_tensor 사용
        x = x + x.new_tensor(self.offset).unsqueeze(0)

        return self.embedding(x)

    def init_weight(self):
        nn.init.xavier_normal_(self.embedding.weight.data)


class FeaturesLinear(nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = FeaturesEmbedding(field_dims, embedding_dim=output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """

        :param x: Long Tensor, size: (batch_size, num_fields)
        :return: Float Tensor, size: (batch_size, output_dim=1)
        """
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationMachine(nn.Module):
    def __init__(self, field_dims, embedding_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embedding_dim)

    def forward(self, x):
        """

        :param x: Long Tensor, size: (batch_size, num_fields)
        :return: Float Tensor, size: (batch_size, 1)
        """
        x = self.embedding(x) # (batch_size, num_fields, embedding_dim)
        square_of_sum = torch.sum(x, dim=1)**2 # (batch_size, embedding_dim)
        sum_of_square = torch.sum(x**2, dim=1) # (batch_size, embedding_dim)
        return 1/2*torch.sum(square_of_sum-sum_of_square, dim=1, keepdim=True)
