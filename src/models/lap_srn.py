#!/usr/bin/env python3
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

# The util functions are stolen from 
#  https://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
# via https://github.com/EdwardTyantov/LapSRN,
def upsample_filter(size):
    """ 
    Create a 2D bilinear upsampling kernel for size (h,w)
    """
    factor = (size + 1)//2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    
def bilinear_upsample_matrix(filter_size, weights):
    """
    Generate weight matrix for bilinear upsampling with a transposed convolution.
    """
    filters_out = weights.size(0)
    filters_in = weights.size(1)
    weights = np.zeros((filters_out, filters_in, 4, 4), dtype=np.float32)
    
    kernel = upsample_filter(filter_size)
    
    for i in range(filters_out):
        for j in range(filters_in):
            weights[i, j, :, :] = kernel
    
    return torch.Tensor(weights)


class ResizeConvolution(nn.Module):
    def __init__(self, num_channels=64):
        super().__init__()
        self.resize = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad = nn.ReflectionPad2d(1)
        self.convolution = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=0)
    
    def forward(self, x):
        x = self.resize(x)
        x = self.pad(x)
        return self.convolution(x)
    
class FeatureExtraction(nn.Module):
    def __init__(self, depth=3, residual=False, upsample=True):
        super().__init__()
        
        LReLu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        filters = nn.Sequential()
        
        for i in range(depth):
                filters.add_module('conv{}'.format(i), nn.Conv2d(64, 64, 3, stride=1, padding=1))
                filters.add_module('lrelu{}'.format(i), LReLu)

        if upsample:
            filters.add_module('resize_conv', ResizeConvolution(num_channels=64))
            filters.add_module('lrelu_upsample', LReLu)                   
        
        self.seq = filters
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.kaiming_normal(m.weight, a=0.2)
                init.xavier_normal(m.weight)
                m.bias.data.fill_(0.0)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(bilinear_upsample_matrix(4, m.weight.data))
        
    def forward(self, x):
        out = self.seq(x)
        return out

    
class ImageReconstruction(nn.Module):
    def __init__(self, upsample=True):
        super(ImageReconstruction, self).__init__()
        self.conv_residual = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        if upsample:
            self.upsample = ResizeConvolution(num_channels=3)
        else:
            self.upsample = lambda x: x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight)
                m.bias.data.fill_(0.0)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(bilinear_upsample_matrix(4, m.weight.data))
        
    def forward(self, lr_image, hr_filter):
        upsampled = self.upsample(lr_image)
        residual = self.conv_residual(hr_filter)
        hr = upsampled + residual
        return hr

    
class LapSRN(nn.Module):
    def __init__(self, depth=10, upsample=True):
        super(LapSRN, self).__init__()
        
        self.in_conv = nn.Sequential((nn.Conv2d(3, 64, 3, stride=1, padding=1)), nn.LeakyReLU(negative_slope=0.2, inplace=True))

        init.xavier_normal(self.in_conv[0].weight)
        self.in_conv[0].bias.data.fill_(0.0)
        
        self.feature_extraction0 = FeatureExtraction(depth=depth, 
                                                     upsample=upsample)
        self.feature_extraction1 = FeatureExtraction(depth=depth,
                                                     upsample=upsample)

        self.image_reconstruction0 = ImageReconstruction(upsample=upsample)
        self.image_reconstruction1 = ImageReconstruction(upsample=upsample)

    def forward(self, image):
        features0 = self.feature_extraction0(self.in_conv(image))
        hr2 = self.image_reconstruction0(image, features0)
        
        features1 = self.feature_extraction1(features0)
        hr4 = self.image_reconstruction1(hr2, features1) 
      
        return hr2, hr4
