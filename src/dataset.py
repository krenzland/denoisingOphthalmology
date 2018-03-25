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
        # Image is at least 50% not entirely black
        is_not_black = (np.array(crop).sum(axis=-1) < 90).sum() < (self.size[0]**2)//2
        # Image contains at least one vessel.
        # 1/256 (for crop size 128) has to be marked as vessel
        # threshold is just to reduce random noise
        contains_vessels =  (1.0 * (np.array(vessels) > 200)).sum() > (64)
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
        max_tries = 5
        for it in range(max_tries):
            # Just trust the user that all imgs have the same size...
            i, j, h, w = self.get_params(imgs[0], self.size)
            crop = imgs[0].crop((j, i, j + w, i + h))
            crop_vessels = vessels.crop((j, i, j + w, i + h))
            if self._is_good_crop(crop, crop_vessels) or it == (max_tries - 1):
                crops = [crop] + [img.crop((j, i, j + w, i + h)) for img in imgs[1:]]
                return crops

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

def load_image(path):
    img = Image.open(path)
    img.load()
    return img 

class Dataset(data.Dataset):
    def __init__(self, path, hr_transform, lr_transform, verbose=False, seed=19534, use_saliency=True):
        super(Dataset, self).__init__
        
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
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
        imgs += self.lr_transform(hr)
        if self.use_saliency:
            saliencies += self.lr_transform(saliency, is_saliency=True)

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

class LrTransform(object):
    def __init__(self, crop_size, factors, max_blur):
        self.crop_size = crop_size
        self.factors = factors
        self.max_blur = max_blur

    def _get_blurs(self):
        if self.max_blur < 0:
            # no blur needed.
            return [0.0] * len(self.factors)

        # First compute total blur
        total_blur = np.random.uniform(0,self.max_blur)

        if len(self.factors) == 1:
            return [total_blur]
        elif len(self.factors) == 2:
            # Blurring twice results in total blur of
            # sqrt(blur_1**2 + blur_2**2)
            # can be derived by conv. of two Gaussians
            # In our case, we want to apply half of the blur to the first stage:
            half_blur = (total_blur**2/2)**0.5
            return [half_blur, total_blur]
        else:
            # Not needed here.
            raise ValueError("More than 2 downsizing ops. not supported")

    def __call__(self, img, is_saliency=False):
        lr_imgs = []
        # LR image needs no saliency and saliency must not be blurred!
        blurs = self._get_blurs() if not is_saliency else [0.0] * (len(self.factors) - 1)
        for factor, blur in zip(self.factors, blurs):
            # First blur
            if blur > 0:
                lr_img = img.filter(ImageFilter.GaussianBlur(blur))
            else:
                lr_img = img
            # Then downsample, if factor != 1
            if factor != 1:
                lr_img = lr_img.resize((self.crop_size//factor, self.crop_size//factor),
                                       Image.BICUBIC)
            lr_imgs.append(lr_img)
        return lr_imgs


        
