import torch
import torch.nn as nn
import torch.nn.functional as F


class MCT(nn.Module):
    def __init__(self, hidden_dim):
        self.fc1 = nn.Linear(hidden_dim, 60)
        self.fc2 = nn.Linear(60, 24)
        self.fc3 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return F.log_softmax(x, dim=1)