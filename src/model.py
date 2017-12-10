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
    
class FeatureExtraction(nn.Module):
    def __init__(self, level, depth=3):
        super(FeatureExtraction, self).__init__()
        
        LReLu = nn.LeakyReLU(negative_slope=0.2)
        filters = nn.Sequential()
        
        # First layer is connected to input image, we only use Y color channel!
        if level == 0:
            filters.add_module(f'conv_input', nn.Conv2d(1, 64, 3, stride=1, padding=1))
            filters.add_module(f'lrelu_input', LReLu)

        for i in range(depth):
            filters.add_module(f'conv{i}', nn.Conv2d(64, 64, 3, stride=1, padding=1))
            filters.add_module(f'lrelu{i}', LReLu)                               

        filters.add_module('convt_upsample', nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1))
        filters.add_module('lrelu_upsame', LReLu)                   
        
        self.seq = filters
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform(m.weight, a=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(bilinear_upsample_matrix(4, m.weight.data))
        
    def forward(self, x):
        out = self.seq(x)
        return out
    
class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.conv_residual = nn.Conv2d(64, 1, 3, stride=1, padding=1) # last filter -> res
        self.upsample = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)

        self.upsample.weight.data.copy_(bilinear_upsample_matrix(4, self.upsample.weight))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, a=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(bilinear_upsample_matrix(4, m.weight.data))
        
    def forward(self, lr_image, hr_filter):
        upsampled = self.upsample(lr_image)
        residual = self.conv_residual(hr_filter)
        return upsampled + residual
    
class LapSRN(nn.Module):
    def __init__(self, upsampling_factor=4, depth=5):
        super(LapSRN, self).__init__()
        
        # TODO: Don't ignore this!
        n_layers = np.log2(upsampling_factor).astype(np.int32)
        
        # Pytorch doesn't consider layers stored in a list.
        # See: https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219
        # As a work-around create layers for 8x-upsampling and ignore unneeded layers
        # TODO: Use http://pytorch.org/docs/0.3.0/nn.html#modulelist

        self.feature_extraction0 = FeatureExtraction(level=0, depth=depth)
        self.feature_extraction1 = FeatureExtraction(level=1, depth=depth)
        self.feature_extraction2 = FeatureExtraction(level=2, depth=depth)
        
        self.image_reconstruction0 = ImageReconstruction()
        self.image_reconstruction1 = ImageReconstruction()
        self.image_reconstruction2 = ImageReconstruction()

    def forward(self, image):
        features0 = self.feature_extraction0(image)
        hr2 = self.image_reconstruction0(image, features0)
        
        features1 = self.feature_extraction1(features0)
        hr4 = self.image_reconstruction1(hr2, features1) 
      
        features2 = self.feature_extraction2(features1)
        hr8 = self.image_reconstruction2(hr4, features2)

        return hr2, hr4, hr8
