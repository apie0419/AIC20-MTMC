import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class MCT(nn.Module):
    def __init__(self, appearance_dim, physic_dim):
        super(MCT, self).__init__()

        self.appearance_dim = appearance_dim
        self.physic_dim = physic_dim
        self.fc1 = nn.Linear(appearance_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc11 = nn.Linear(physic_dim, 5)
        self.bn11 = nn.BatchNorm1d(5)
        self.fc22 = nn.Linear(5, 2)
        self.bn22 = nn.BatchNorm1d(2)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 2)
        self.bn5 = nn.BatchNorm1d(2)
        self.fc6 = nn.Linear(4, 2)

        self.fc1.apply(weights_init_kaiming)
        self.bn1.apply(weights_init_kaiming)
        self.fc11.apply(weights_init_kaiming)
        self.bn11.apply(weights_init_kaiming)
        self.fc22.apply(weights_init_kaiming)
        self.bn22.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)
        self.fc3.apply(weights_init_kaiming)
        self.bn3.apply(weights_init_kaiming)
        self.fc4.apply(weights_init_kaiming)
        self.bn4.apply(weights_init_kaiming)
        self.fc5.apply(weights_init_kaiming)
        self.bn5.apply(weights_init_kaiming)
        self.fc6.apply(weights_init_kaiming)


    def forward(self, x):

        x, x_2 = x[:, :self.appearance_dim], x[:, self.appearance_dim:]
        
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x_2 = self.fc11(x_2)
        x_2 = F.relu(self.bn11(x_2))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x_2 = self.fc22(x_2)
        x_2 = F.relu(self.bn22(x_2))
        x = self.fc3(x)

        x = F.relu(self.bn3(x))
        x = self.fc4(x)
        x = F.relu(self.bn4(x))
        x = self.fc5(x)
        x = F.relu(self.bn5(x))
        x = self.fc6(torch.cat((x, x_2), 1))
        # x = F.relu(x)

        return x
