#!/usr/bin/env python3
import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Eps is a relaxation parameter, the paper sets it to 1e-3.
        """
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        
    def forward(self, x, y):
        d = torch.add(x, -y)
        e = torch.sqrt(d**2 + self.eps)
        return torch.mean(e)
