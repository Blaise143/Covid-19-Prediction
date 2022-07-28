import pandas as pd
import numpy as np
import torch, torchvision
from torch import nn
import torch.nn.functional as F

class BlaiseNet(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 2) -> None:
        '''
        :param num_hidden: Number of hidden layers
        :param in_channels: Number of channels
        '''
        super(BlaiseNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.Conv2d(kernel_size=7, in_channels=64, out_channels=128, padding=0, stride=3),
            nn.ReLU(),
            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=256, padding=0, stride=3),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.Linear(in_features=128, out_features=68),
            nn.Linear(in_features=68, out_features=num_classes)
        )

    def forward(self, x):
        '''
        Emplements forward prop
        :param x: Image
        :return: x
        '''
        x = self.convs(x)
        x = self.layers(x)
        return x
model = BlaiseNet()
print(model)