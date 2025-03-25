import torch
import torch.nn as nn
import torch.nn.functional as F

class Mask(nn.Module):
    def __init__(self, *, channel=1, height=513, width=1024, device='cuda'):
        super().__init__()
        self.weight = nn.Parameter(torch.full((channel, height, width), 3.0))
        # self.weight = nn.Parameter(torch.randn((channel, height, width)))
        self.device = device
        self.to(device)
        
    def forward(self):
        return torch.sigmoid(self.weight)
    
class Multi_Class_Mask(nn.Module):
    def __init__(self, *, num_classes=2, height=513, width=1024, device='cuda'):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((num_classes, height, width)))
        self.device = device
        self.to(device)

    def forward(self):
        # (num_classes, H, W) → softmax over channel dimension (dim=0) at each (h,w)
        return F.softmax(self.weight, dim=0)