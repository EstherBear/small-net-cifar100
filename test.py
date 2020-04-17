import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import os
import numpy as np
class SE(nn.Module):
    def __init__(self, inchannels, se_ratio):
        super().__init__()
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(inchannels, int(inchannels/se_ratio)),
            nn.ReLU(),
            nn.Linear(int(inchannels / se_ratio), inchannels)
        )

    def forward(self, x):
        out = self.AvgPool(x)
        out = out.view(x.size(0), -1)
        out = self.SEblock(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        print(x)
        print(out)
        return x * torch.sigmoid(out)


x = torch.from_numpy(np.arange(24).reshape((2, 3, 2, 2))).float()
net = SE(3, 1)
net(x)

'''
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
])

traindata = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)

trainloader = DataLoader(traindata, batch_size=128, shuffle=True, num_workers=2)

print(len(trainloader))
print(len(trainloader.dataset))
'''