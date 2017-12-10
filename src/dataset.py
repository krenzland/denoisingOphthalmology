#!/usr/bin/env python3
import os

import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import Compose, RandomCrop, ToTensor, Scale, RandomHorizontalFlip

class RandomRotation(object):
    def __init__(self, angles=None):
        if angles:
            self.angles = angles
        else:
            self.angles = np.array([90, 180, 270])

    @staticmethod
    def get_params(angles):
        angle = np.random.choice(angles)

        return angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.angles)

        return img.rotate(angle)

def load_image(path):
    # We only use the Y channel for the upscaling.
    img = Image.open(path).convert('YCbCr')
    return img.split()[0]

# Utility function, useful for default values.
id_ = lambda x: x

class Dataset(data.Dataset):
    def __init__(self, path, hr_transform=id_, lr_transforms=[id_,]*3):
        super(Dataset, self).__init__
        
        self.hr_transform = hr_transform
        self.lr_transforms = lr_transforms
        
        image_suffixes = ['jpg', 'tif', 'ppm']        
        full_filename = lambda f: os.path.abspath(os.path.join(path, f))
        is_image = lambda f: any([f.endswith(f'{suff}') for suff in image_suffixes])
        
        self.filenames = [full_filename(f) for f in os.listdir(path) if is_image(f)]
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img = load_image(self.filenames[idx])
        
        # Compute all transformations for the downscaled versions.
        hr = self.hr_transform(img)
        lrs = [lr_transform(hr) for lr_transform in self.lr_transforms]
        
        to_tensor = ToTensor()
        tensors = [to_tensor(i) for i in (lrs + [hr])]
        # lr, hr2, hr4, hr8 = ...
        # shape for crops of size 100: 12, 25, 50, 100
        return tensors
    
def get_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        # RandomScale between 0.5 and 1.0
        RandomRotation(),
        RandomHorizontalFlip()
        # RandomVerticalFlip
    ])

def get_lr_transform(crop_size, factor):
    # Factor = 2 for hr4, Factor = 4 for hr2
    return Scale(crop_size//factor)

# ALL IMAGES ARE TRANSFORMED BY: 1) Rand. scaling between 0.5 and 1.0
# 2) Rotation by 90, 180, 270° 3) Flip img. hor./vert with prob of 0.5
