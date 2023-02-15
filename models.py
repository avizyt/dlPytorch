import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self, in_feature, n_classes, batch_size) -> None:
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_feature, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        n_ch = self.encoder(torch.empty(batch_size, in_feature, 28, 28)).size(-1)
        

        self.decoder = nn.Sequential(
            nn.Linear(n_ch, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        out = self.decoder(features)
        out = F.log_softmax(out, dim=1)
        return out 
    
if __name__ == "__main__":

    model = Net(1, 10, 32).to(device='cuda')
    summary(model, (1, 28, 28), 32)
    # print(model)