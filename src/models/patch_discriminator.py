#!/usr/bin/env python3
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

class PatchD(nn.Module):
    def __init__(self, num_layers = 4, num_channels=3, use_sigmoid=False):
        """
        Patch discriminator.
        From Image-to-Image Translation with Conditional Adversarial Networks,
        Isola et al
        
        num_layers | patch-size
        2          |  16 x  16
        4          |  70 x  70
        6          | 574 x 574
        """        
        super().__init__()
        
        # Layer size:
        # 64 -> 128 -> 256 -> 512 -> 512 -> ...
        n_filters = lambda layer_idx: min(512, 2**(layer_idx+6))
            
        def make_layer(layer_idx):
            """
            Makes one convolution layer.
            If layer_idx it consists of Conv2D -> InstanceNorm -> LRelu, else of
            Conv2D -> LRelu
            layer_idx: Starts at 0
            """
            in_channels = num_channels if layer_idx == 0 else n_filters(layer_idx - 1)
            out_channels = n_filters(layer_idx)
            
            stride = 1 if layer_idx == (num_layers - 1) else 2
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride,
                            padding=1)

            # Paper used Batch-Norm.
            # Instance norm from "Instance Normalization - The Missing Ingredient for Fast Stylization",
            # Ulyanov, Vedali
            norm = nn.InstanceNorm2d(out_channels)
            relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            if layer_idx == 0:
                return [conv, relu]
            else:
                return [conv, norm, relu]
        
        layers = []
        
        # Build up downsampling conv. layers.
        for idx in range(num_layers):
            layers.extend(make_layer(idx))
        
        # Map to 1D-Output.
        last_outsize = n_filters(num_layers - 1)
        layers += [
            nn.Conv2d(in_channels=last_outsize, out_channels=1, kernel_size=4, stride=1, padding=1)
        ]
        
        if use_sigmoid:
            self.out = nn.Sigmoid()
        else:
            self.out = lambda x: x
            
        self.layers = nn.Sequential(*layers)            
        
        # Init all weights to a small value.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0.0)

    def forward(self, x, use_sigmoid):
        # Return one number per input image.
        x = self.layers(x)
        if use_sigmoid:
            x = self.out(x)
        return x.view(x.size(0),-1)
