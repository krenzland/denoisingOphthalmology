#!/usr/bin/env python3
import os
from enum import Enum
import numbers
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageFile
import torch.utils.data as data
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, Lambda, CenterCrop, Grayscale
import torchvision.transforms.functional as F

# See: https://github.com/keras-team/keras/issues/5475
# Only affects validation set!
ImageFile.LOAD_TRUNCATED_IMAGES = True

class RandomScaling(object):
    """Resize the input PIL Image by a factor of (0.5, 1.0)
    """

    def __init__(self, crop_size, interpolation=Image.BICUBIC):
        self.crop_size = crop_size
        self.interpolation = interpolation

    @staticmethod
    def get_params():
        return np.random.uniform(0.5, 1.0)
        
    def __call__(self, img, vessels=None):
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

        img = F.resize(img, new_size, self.interpolation)

        if vessels is not None:
            vessels = F.resize(img, new_size, self.interpolation)
            return img, vessels
        else:
            return img

class RandomRotation(object):
    def __init__(self, angles=np.array([0, 90, 180, 270])):
        self.angles = angles

    @staticmethod
    def get_params(angles):
        angle = np.random.choice(angles)

        return angle

    def __call__(self, img, vessels):
        """
        Args:
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        angle = self.get_params(self.angles)
        img = img.rotate(angle, expand=True)
        if vessels is not None:
            vessels = vessels.rotate(angle, expand=True) 
            return img, vessels
        else:
            return img

class RandomFlip(object):
    def __call__(self, img, vessels):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        flip_vertical = random.random() < 0.5
        flip_horicontal = random.random() < 0.5

        if flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            vessels = vessels.transpose(Image.FLIP_TOP_BOTTOM)

        if flip_horicontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            vessels = vessels.transpose(Image.FLIP_LEFT_RIGHT)

        return img, vessels

    def __repr__(self):
        return self.__class__.__name__ + '()'

class SmartRandomCrop(object):
    """Copied from torchvision RandomCrop"""
    def __init__(self, size, padding=0):
        self.grayscale = Grayscale()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def _is_good_crop(self, crop, vessels):
        is_not_black = (1.0*(np.array(self.grayscale(crop))) < 30).sum() < (self.size[0]**2)//2
        contains_vessels =  (1.0 * (np.array(vessels) > 80)).sum() > (128*20)
        return is_not_black and contains_vessels
    
    def __call__(self, img, vessels):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        crop = None
        max_tries = 20
        for it in range(max_tries):
            i, j, h, w = self.get_params(img, self.size)
            crop = img.crop((j, i, j + w, i + h))
            crop_vessels = vessels.crop((j, i, j + w, i + h))
            if self._is_good_crop(crop, crop_vessels):
                return crop
        # Giving up
        return crop        

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

def load_image(path):
    # We only use the Y channel for the upscaling.
    # img = Image.open(path).convert('YCbCr')
    # return img.split()[0]
    img = Image.open(path)
    img.load()
    return img 

class Dataset(data.Dataset):
    def __init__(self, path, hr_transform, lr_transforms, verbose=False, seed=19534):
        super(Dataset, self).__init__
        
        self.hr_transform = hr_transform
        self.lr_transforms = lr_transforms
        self.verbose = verbose
        
        image_suffixes = ['jpg', 'jpeg', 'tif', 'ppm', 'png']        
        full_filename = lambda f: os.path.abspath(os.path.join(path, f))
        is_image = lambda f: any([f.endswith(suff) for suff in image_suffixes])
        
        self.filenames = [full_filename(f) for f in os.listdir(path) if is_image(f)]
        random.Random(seed).shuffle(self.filenames)

        self.images = []
        self.vessels = []
        
        for i,f in enumerate(self.filenames):
            self.images.append(load_image(f))

            # Load vessel data
            f = Path(f)
            vessel_file_name = f.parent / 'vessels' / (f.stem + '.jpg') 
            self.vessels.append(load_image(str(vessel_file_name)).convert('YCbCr').split()[0])
            if verbose and i % 50 == 0:
                print("Dataset loading, {} out of {} images read!".format(i, len(self.filenames))) 
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        vessels = self.vessels[idx]
        
        # Compute all transformations for the downscaled versions.
        hr = self.hr_transform(img, vessels)
        
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
                raise ValueError('{} is not a valid ratio!'.format(split_ratio))
                
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

class HrTransform(object):
    def __init__(self, crop_size, random=True):
        self.crop_size = crop_size
        if random:
            self.random_scaling = RandomScaling(crop_size)
            self.random_rotation = RandomRotation()
            self.random_flip = RandomFlip()
            self.crop = SmartRandomCrop(crop_size)
        else:
            self.crop = CenterCrop(crop_size)
        self.random = random

    def __call__(self, img, vessels):
        if self.random:
            img, vessels = self.random_scaling(img, vessels)
            img, vessels = self.random_rotation(img, vessels)
            img, vessels = self.random_flip(img, vessels)
            return self.crop(img, vessels)
        else:
            return self.crop(img) 
        

def get_hr_transform(crop_size, random=True):
    return HrTransform(crop_size, random=random)

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
