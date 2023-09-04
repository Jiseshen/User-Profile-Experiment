import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, hidden, class_num):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, hidden)
        self.linear2 = nn.Linear(hidden, class_num)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x