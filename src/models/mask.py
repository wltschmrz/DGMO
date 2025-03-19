import torch
import torch.nn as nn

class Mask(nn.Module):
    def __init__(self, *, channel=1, height=513, width=1024, device='cuda'):
        super().__init__()
        # self.weight = nn.Parameter(torch.full((channel, height, width), 3.0))
        self.weight = nn.Parameter(torch.randn((channel, height, width)))
        self.device = device
        self.to(device)
        
    def forward(self):
        return torch.sigmoid(self.weight)