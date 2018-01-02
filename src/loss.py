#!/usr/bin/env python3
from collections import namedtuple

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        """
        Eps is a relaxation parameter, the paper sets it to 1e-3.
        """
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        
    def forward(self, x, y):
        d = torch.add(x, -y)
        e = torch.sqrt(d**2 + self.eps)
        return torch.mean(e)

# Specifies which layers we want to use for our loss
LossOutput = namedtuple("ContentLoss", ['pool_1', 'pool_2'])

# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg):
        super(LossNetwork, self).__init__()
        # Vgg has to be the VGG16 model from torchvision.
        assert(isinstance(vgg, VGG) and len(vgg.features) == 31)

        # Maps layer-id -> layer_name
        self.layer_map = {
            '4': 'pool_1',
            '9': 'pool_2'
        }
        # Drop all layers that are not needed from the model.
        last_layer = max([int(i) for i in self.layer_map])
        self.vgg_layers = nn.Sequential(*list(vgg.features)[:last_layer+1])
        
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_map:
                output[self.layer_map[name]] = x
        return LossOutput(**output)

class PerceptualLoss(nn.Module):
    def __init__(self, criterion, loss_network):
        super(PerceptualLoss, self).__init__()
        self.loss_network = loss_network
        self.criterion = criterion
        
    def forward(self, x, y):
        # x/y only have the y-channel, vgg expects RGB values.
        # Just use y as r/g/b each.
        featX = self.loss_network(torch.cat((x,x,x), 1))
        featY = self.loss_network(torch.cat((y,y,y), 1))
        
        content_loss = 0.0
        for a, b in zip(featX, featY):
            content_loss += self.criterion(a, b)
            
        return content_loss

def make_vgg16_loss(criterion):
    vgg = vgg16(pretrained=True)
    loss_network = LossNetwork(vgg).eval()
    del vgg
    ploss = PerceptualLoss(criterion, loss_network)
    return ploss

# Layers of VGG16:
# Sequential(
#   (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU(inplace)
#   (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU(inplace)
#   (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
#   (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (6): ReLU(inplace)
#   (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU(inplace)
#   (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
#   (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace)
#   (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU(inplace)
#   (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU(inplace)
#   (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
#   (17): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (18): ReLU(inplace)
#   (19): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (20): ReLU(inplace)
#   (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (22): ReLU(inplace)
#   (23): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
#   (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (25): ReLU(inplace)
#   (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (27): ReLU(inplace)
#   (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (29): ReLU(inplace)
#   (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
# )
