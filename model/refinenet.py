import torch
import torch.nn as nn


class RefineNet_base(nn.Module):
    def __init__(self, in_dim=75, out_dim=45, flatten_size=1):
        super(RefineNet_base, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, 160*flatten_size),
                                    nn.BatchNorm1d(160*flatten_size), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(160*flatten_size, 256*flatten_size),
                                    nn.BatchNorm1d(256*flatten_size), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(256*flatten_size, 256*flatten_size),
                                    nn.BatchNorm1d(256*flatten_size), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(256*flatten_size, 128*flatten_size),
                                    nn.BatchNorm1d(128*flatten_size), nn.ReLU())
        self.layer5 = nn.Linear(128*flatten_size, out_dim)
        self.out_dim = out_dim

    def forward(self, input_x):
        x = self.layer1(input_x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = self.layer5(x)

        return output


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.block = RefineNet_base()

    def forward(self, input_x):
        # 2d pose + root-relative 3d pose --> refined root-relative 3d pose
        output = self.block(input_x)
        return output

