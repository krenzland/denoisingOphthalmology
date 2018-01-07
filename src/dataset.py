#!/usr/bin/env python3
import os

import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, Lambda
import torchvision.transforms.functional as F

class RandomScaling(object):
    """Resize the input PIL Image by a factor of (0.5, 1.0)
    """

    def __init__(self,interpolation=Image.BILINEAR):
        self.interpolation = interpolation

    @staticmethod
    def get_params():
        return np.random.uniform(0.5, 1.0)
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled by random factor \in (0.5, 1.0).

        Returns:
            PIL Image: Rescaled image.
        """
        s = self.get_params()
        oW, oH = img.size
        nW = int(s * oW)
        nH = int(s * oH)
        # Make sure we are not smaller than the cropsize!
        nW = max(self.crop_size, int(s * oW))
        nH = max(self.crop_size, int(s * oH))
        new_size = nW, nH
        return F.resize(img, new_size, self.interpolation)

class RandomRotation(object):
    def __init__(self, angles=np.array([0, 90, 180, 270])):
        self.angles = angles

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
        RandomScaling(),
        RandomRotation(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomCrop(crop_size)
    ])

def get_lr_transform(crop_size, factor):
    # Factor = 2 for hr4, Factor = 4 for hr2
    return Resize(crop_size//factor)
