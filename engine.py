import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import wandb

import matplotlib.pyplot as plt
import numpy as np
from models import Net
from utils import train, test

wandb.init(
    project="CIFER-clf"
)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [transforms.ToTensor()])

training_data = FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_data = FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

batch_size = 32

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


in_feature = 3
n_classes = 10
lr = 0.001
model1 = Net(in_feature=in_feature, n_classes=n_classes, batch_size=batch_size).to(device)

optimizer = torch.optim.Adam(model1.parameters(), lr=lr)
epochs=30




wandb.watch(model1, log='all')
for epoch in range(epochs):
    train(model1, device, train_loader, optimizer, epoch, steps_per_epoch=60000//batch_size)
    test(model1, device, test_loader, classes=n_classes)

print("Finished Training")