import os
from enum import Enum
import random
from pathlib import Path
import numpy as np
import torch.utils.data as data
from torchvision.transforms import ToTensor, Normalize
from PIL import Image, ImageFile

# See: https://github.com/keras-team/keras/issues/5475
# Only affects validation set!
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        if self.use_saliency:
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

        return (tensors, [])
    
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
