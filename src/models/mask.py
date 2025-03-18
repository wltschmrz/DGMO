import torch
import torch.nn as nn

class Mask(nn.Module):
    def __init__(self, device, channel=1, height=1024, width=513):
        super().__init__()
        # self.weight = nn.Parameter(torch.randn((channel, height, width)))
        self.weight = nn.Parameter(torch.full((channel, height, width), 3.0))
        self.to(device)
        self.device = device
    def forward(self):
        return torch.sigmoid(self.weight)