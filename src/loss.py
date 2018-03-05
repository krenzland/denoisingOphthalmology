#!/usr/bin/env python3
from collections import namedtuple

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG
from torchvision.transforms import Normalize

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

# Specifies which layers we want to use for our loss
LossOutput = namedtuple("ContentLoss", ['pool_1', 'pool_4'])

# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg, layer_map):
        super(LossNetwork, self).__init__()
        # Vgg has to be the VGG16 model from torchvision.
        assert(isinstance(vgg, VGG) and len(vgg.features) == 31)

        # Maps layer-id -> layer_name
        self.layer_map = layer_map
        # Drop all layers that are not needed from the model.
        last_layer = max([int(i) for i in self.layer_map])
        self.vgg_layers = nn.Sequential(*list(vgg.features)[:last_layer+1])

        # Deactivate gradient computation
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_map:
                layer_name = self.layer_map[name]
                output[layer_name] = x
        return LossOutput(**output)

class PerceptualLoss(nn.Module):
    def __init__(self, criterion, loss_network, weight_map):
        super(PerceptualLoss, self).__init__()
        self.loss_network = loss_network
        self.criterion = criterion
        self.weight_map = weight_map
 
    def forward(self, x, y):
        featX = self.loss_network(x)
        featY = self.loss_network(y)
        
        content_loss = 0.0
        for layer_name in featX._asdict():
            weight = self.weight_map[layer_name]
            layer_loss = self.criterion(getattr(featX, layer_name), getattr(featY, layer_name))
            content_loss += weight * layer_loss
            
        return content_loss

def make_vgg16_loss(criterion):
    # Key = number in sequential layer of vgg, value = (layer_name, weight)
    layer_map = {
        '4': 'pool_1',
        # '9': 'pool_2'
        '23': 'pool_4'
    }
    weight_map = {
        'pool_1': 1.0,
        'pool_2': 1.0,
        'pool_4': 0.1
    }
    vgg = vgg16(pretrained=True)
    loss_network = LossNetwork(vgg, layer_map=layer_map).eval()
    del vgg
    ploss = PerceptualLoss(criterion, loss_network=loss_network, weight_map=weight_map)
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
