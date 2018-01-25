#!/usr/bin/env python3
import os
from enum import Enum

import numpy as np
from PIL import Image, ImageFilter
import torch.utils.data as data
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, Lambda, CenterCrop
import torchvision.transforms.functional as F

class RandomScaling(object):
    """Resize the input PIL Image by a factor of (0.5, 1.0)
    """

    def __init__(self, crop_size, interpolation=Image.BICUBIC):
        self.crop_size = crop_size
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
        return img.rotate(angle, expand=True)

class SmartRandomCrop(object):
    def __init__(self, crop_size, black_threshold):
        self.random_crop = RandomCrop(crop_size)
        self.black_treshold = black_threshold

    def __call__(self, img):
        crop = None
        for i in range(0, 100):
            crop = self.random_crop(img)

            count_black = (1.0*(np.array(crop) < 10)).sum()
            if count_black < self.black_treshold:
                return crop
        return crop

def load_image(path):
    # We only use the Y channel for the upscaling.
    img = Image.open(path).convert('YCbCr')
    return img.split()[0]

class Dataset(data.Dataset):
    def __init__(self, path, hr_transform, lr_transforms, verbose=False):
        super(Dataset, self).__init__
        
        self.hr_transform = hr_transform
        self.lr_transforms = lr_transforms
        self.verbose = verbose
        
        image_suffixes = ['jpg', 'jpeg', 'tif', 'ppm', 'png']        
        full_filename = lambda f: os.path.abspath(os.path.join(path, f))
        is_image = lambda f: any([f.endswith(f'{suff}') for suff in image_suffixes])
        
        self.filenames = [full_filename(f) for f in os.listdir(path) if is_image(f)]

        self.images = []
        for i,f in enumerate(self.filenames):
            self.images.append(load_image(f))
            if verbose and i % 50 == 0:
                print(f"Dataset loading, {i} out of {len(self.filenames)} images read!") 
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        
        # Compute all transformations for the downscaled versions.
        hr = self.hr_transform(img)
        
        imgs = [hr]
        for transform in self.lr_transforms:
            imgs.append(transform(imgs[-1]))
        
        to_tensor = ToTensor()
        tensors = [to_tensor(i) for i in reversed(imgs)]
        # lr, hr2, hr4, hr8 = ...
        # shape for crops of size 100: 12, 25, 50, 100
        return tensors

    
class Split(Enum):
    ALL = 0,
    TRAIN = 1
    TEST = 2

class SplitDataset(data.Dataset):
    def __init__(self, dataset, split=Split.ALL, split_ratio=None):
        super(SplitDataset, self).__init__
        
        if not isinstance(split, Split):
            raise TypeError()
            
        if split is split.ALL:
            if split_ratio is not None:
                raise ValueError('Split==all implies no ratio')
        else:
            if split_ratio is None or not(0.0 < split_ratio < 1.0):
                raise ValueError(f'{split_ratio} is not a valid ratio!')
                
        self.dataset = dataset
        self.split = split
        self.split_ratio = split_ratio
        
    def __len__(self):
        total_len = len(self.dataset)
        if self.split is Split.TRAIN:
            return int(np.ceil(self.split_ratio * total_len))
        elif self.split is Split.TEST:
            return int((np.floor((1.0 - self.split_ratio) * total_len)))
        else:
            return total_len
    
    def __getitem__(self, idx):
        if idx < 0:
            raise ValueError('Negative idx aren\t supported!')
        if self.split is Split.TEST:
            idx += int(np.ceil(self.split_ratio * len(self.dataset)))    

        return self.dataset[idx]
    
blur_filter = lambda img: img.filter(ImageFilter.GaussianBlur(0 * np.random.uniform(0,1)))
# TODO: Add blur!
blur_fliter = lambda img: img

def get_lr_transform(crop_size, factor, random=True):
    # Factor = 2 for hr4, Factor = 4 for hr2
    # Only blur LR image otherwise we get really noisy output!
    if factor == 4 and not random:
        return Compose([
            #Lambda(blur_filter),
            Resize(crop_size//factor, interpolation=Image.BICUBIC)
        ])
    else:
        return Resize(crop_size//factor, interpolation=Image.BICUBIC)

def get_hr_transform(crop_size, random=True):
    if random:
        return Compose([
            RandomScaling(crop_size),
            RandomRotation(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            #RandomCrop(crop_size),
            SmartRandomCrop(crop_size, (crop_size**2)//2)
        ])
    else:
        return CenterCrop(crop_size)

blur_filter = lambda img: img.filter(ImageFilter.GaussianBlur(0 * np.random.uniform(0,1)))
# TODO: Add blur!
blur_fliter = lambda img: img

def get_lr_transform(crop_size, factor, random=True):
    # Factor = 2 for hr4, Factor = 4 for hr2
    # Only blur LR image otherwise we get really noisy output!
    if factor == 4 and not random:
        return Compose([
            #Lambda(blur_filter),
            Resize(crop_size//factor, interpolation=Image.BICUBIC)
        ])
    else:
        return Resize(crop_size//factor, interpolation=Image.BICUBIC) 
