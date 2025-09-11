from torch import nn as nn


class LisaCNN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (8, 8), stride=(2, 2), padding=3)
        self.conv2 = nn.Conv2d(64, 128, (6, 6), stride=(2, 2), padding=0)
        self.conv3 = nn.Conv2d(128, 128, (5, 5), stride=(1, 1), padding=0)
        self.fc = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(nn.ReLU()(x))
        x = self.conv3(nn.ReLU()(x))
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
