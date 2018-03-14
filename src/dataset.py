#!/usr/bin/env python3
import os
from enum import Enum
import numbers
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageFile
import torch.utils.data as data
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, Lambda, CenterCrop, Grayscale, ColorJitter, Normalize
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
        
    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be scaled by random factor \in (0.5, 1.0).

        Returns:
            PIL Image: Rescaled image.
        """
        s = self.get_params()
        # Just trust the user that all imgs have the same size...
        oW, oH = imgs[0].size

        # Make sure we are not smaller than the cropsize!
        s = max(s, max(self.crop_size/oW, self.crop_size/oH))
        
        nW = int(s * oW)
        nH = int(s * oH)
        new_size = nW, nH

        imgs = [F.resize(img, new_size, self.interpolation) for img in imgs]

        return imgs

class RandomRotation(object):
    def __init__(self, angles=np.array([0, 90, 180, 270])):
        self.angles = angles

    @staticmethod
    def get_params(angles):
        angle = np.random.choice(angles)
        return angle

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        angle = self.get_params(self.angles)
        return [img.rotate(angle, expand=True) for img in imgs]

class RandomFlip(object):
    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        flip_vertical = random.random() < 0.5
        flip_horicontal = random.random() < 0.5

        if flip_vertical:
            imgs = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in imgs]

        if flip_horicontal:
            imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]

        return imgs

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
    
    def __call__(self, imgs, vessels=None):
        """
        Args:
            imgs (PIL Images): Image to be cropped.
            First image in imgs is used to determine quality of crop!
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            imgs = [F.pad(img, self.padding) for img in imgs]

        crops = [None] * len(imgs)
        max_tries = 20
        for it in range(max_tries):
            # Just trust the user that all imgs have the same size...
            i, j, h, w = self.get_params(imgs[0], self.size)
            crop = imgs[0].crop((j, i, j + w, i + h))
            crop_vessels = vessels.crop((j, i, j + w, i + h))
            if self._is_good_crop(crop, crop_vessels) or it == (max_tries - 1):
                crops = [crop] + [img.crop((j, i, j + w, i + h)) for img in imgs[:1]]
                return crops

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

def load_image(path):
    img = Image.open(path)
    img.load()
    return img 

class Dataset(data.Dataset):
    def __init__(self, path, hr_transform, lr_transforms, verbose=False, seed=19534, use_saliency=True):
        super(Dataset, self).__init__
        
        self.hr_transform = hr_transform
        self.lr_transforms = lr_transforms
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        self.verbose = verbose
        
        image_suffixes = ['jpg', 'jpeg', 'tif', 'ppm', 'png']        
        full_filename = lambda f: os.path.abspath(os.path.join(path, f))
        is_image = lambda f: any([f.endswith(suff) for suff in image_suffixes])
        
        self.filenames = [full_filename(f) for f in os.listdir(path) if is_image(f)]
        random.Random(seed).shuffle(self.filenames)

        self.images = []
        self.vessels = []
        self.use_saliency = use_saliency
        if use_saliency:
            self.saliency = []
        
        for i,f in enumerate(self.filenames):
            self.images.append(load_image(f))

            # Load vessel data
            f = Path(f)
            vessel_file_name = f.parent / 'vessels' / (f.stem + '.jpg') 
            self.vessels.append(load_image(str(vessel_file_name)).convert('YCbCr').split()[0])

            if use_saliency:
                saliency_file_name = f.parent / 'saliency' / (f.stem + '.jpg') 
                self.saliency.append(load_image(str(saliency_file_name)).convert('YCbCr').split()[0])

            if verbose and i % 50 == 0:
                print("Dataset loading, {} out of {} images read!".format(i, len(self.filenames))) 
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        vessels = self.vessels[idx]
        saliency = self.saliency[idx]
        
        # Compute all transformations for the downscaled versions.
        
        if self.use_saliency:
            hr, saliency = self.hr_transform(img, vessels=vessels, saliency=saliency)
            saliencies = [saliency]
        else:
            hr = self.hr_transform(img, vessels=vessels)
        
        imgs = [hr]
        for transform in self.lr_transforms:
                imgs.append(transform(imgs[-1]))
        if self.use_saliency:
            # 1: because we don't need a saliency map for the lr image!
            for transform in self.lr_transforms[:-1]:
                # TODO: Don't do this if transform contains blur!
                saliencies.append(transform(saliencies[-1]))
        
        # lr, hr2, hr4, hr8 = ...
        # shape for crops of size 100: 12, 25, 50, 100
        tensors = [self.normalize(self.to_tensor(i)) for i in reversed(imgs)]
        if self.use_saliency:
            tensors_saliency = [self.to_tensor(i) for i in reversed(saliencies)]
            return (tensors, tensors_saliency)

        return (tensors)
    
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

    def __call__(self, img, vessels, saliency=None):
        if self.random:
            if saliency is not None:
                # Needs to be done before cropping
                img, vessels, saliency = self.random_scaling([img, vessels, saliency])
                img, saliency = self.crop([img, saliency], vessels=vessels)
                img, saliency = self.random_rotation([img, saliency])
                img, saliency = self.random_flip([img, saliency])
                return img, saliency
            else:
                img, vessels = self.random_scaling(img, [vessels])
                img = self.crop([img], vessels=vessels)
                img = self.random_rotation([img])
                img = self.random_flip([img])
                return img
        else:
            img = self.crop(img) 
            if saliency is not None:
                saliency = self.crop(saliency)
                return img, saliency
            return img

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

def get_blur_transform(crop_size, blur=1.5, random=True):
    assert(blur > 0.0)
    blur_func = lambda img: img.filter(ImageFilter.GaussianBlur(blur * np.random.uniform(0, blur)))
    return Lambda(blur_func)
