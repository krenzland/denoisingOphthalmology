#!/usr/bin/env python3
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Normalize, CenterCrop
from PIL import Image, ImageFilter
from skimage.filters import frangi, threshold_otsu

#from dataset import Dataset, Split, SplitDataset, get_lr_transform, get_hr_transform
from model import LapSRN

def upsample_tensor(model, lr, return_time=False, to_normalize=False):      
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    normalize = Normalize(mean=mean, std=std)
    to_tensor = ToTensor()
    
    # First create appropiate input image
    lr = to_tensor(lr)
    if to_normalize:
        lr = normalize(lr)
        
    lr = Variable(lr).cuda().unsqueeze(0)     
    def denormalize(out):
        # Undo input[channel] = (input[channel] - mean[channel]) / std[channel]
        for t, m, s in zip(out, mean, std):
            t.mul_(s)
            t.add_(m)
        return out.clamp(0,1)
    
    start = time.perf_counter()
    upscaled = model(lr)
    end = time.perf_counter()
    
    hr2, hr4 = [denormalize(out.squeeze(0).cpu().data) for out in upscaled]

    elapsed_time = end - start

    if return_time:
        return elapsed_time, hr2, hr4
    else:
        return hr2, hr4
    
def upsample_bic(lr):
    lr_size = np.array(lr.size)
    hr2 = lr.resize(2*lr_size, Image.BICUBIC)
    hr4 = lr.resize(4*lr_size, Image.BICUBIC)
    return hr2, hr4

def residual(y, y_hat, transform=lambda x:x):
    y = transform(y)
    y_hat = transform(y_hat)
    return chop.subtract(y, y_hat, scale=0.05)

def acc(y, y_hat, transform=lambda x:x):
    y = np.array(transform(y))
    y_hat = np.array(transform(y_hat))
    total = y.size * 1.0
    correct = ((y == y_hat)*1.0).sum()

    return (correct/total)*100

def mse(y, y_hat, transform=lambda x:x):
    y = np.array(transform(y))
    y_hat = np.array((y_hat))
    
    diff = y_hat/255.0 - y/255.0
    return np.sum(diff**2)/diff.size    

def psnr(y, y_hat, transform=lambda x:x):
    error = mse(y, y_hat, transform)
    psnr = -10 * np.log10(error)
    return psnr

def vessels(img, mask=None):
    img = np.array(img.convert('YCbCr').split()[0]).astype(float)

    # Ignore background
    if mask is None:
        thresh = threshold_otsu(img)
        black = img < thresh
    else:
        black = np.array(mask) == 0

    # Compute vessels
    vessels = frangi(img, scale_range=(2, 6), beta2=3.8, beta1=0.4)

    # Reset black pixels to black. Otherwise we get a surrounding ring.
    vessels[black] = 0.0
    vessels = (vessels > 0.2)*1.0
    vessels = Image.fromarray(vessels*255).convert('RGB')    

    return vessels

def main():
    checkpoint_dir = Path('../best_checkpoints/')
    models = ['l1', 'perceptual', 'perceptual_pool4', 'wgan']
    model_to_checkpoint = lambda m: checkpoint_dir / '{}.pt'.format(m) 

    rows = []
    for model_name in models:
        checkpoint = torch.load(model_to_checkpoint(model_name))

        model = LapSRN(depth=10).cuda().eval()
        model.load_state_dict(checkpoint['model_state'])

        drive_dir = Path('../data/raw/DRIVE/training/') 
        drive_range = range(21,41)

        for num in drive_range:
            print("Analysing image {} for model {}.".format(num, model_name))
            # Load one drive image with mask and ground truth segmentation.
            mask = Image.open(drive_dir / 'mask/{}_training_mask.gif'.format(num))
            img = Image.open(drive_dir / 'images/{}_training.tif'.format(num))
            gt = Image.open(drive_dir / '1st_manual/{}_manual1.gif'.format(num)).convert('YCbCr').split()[0]

            to_pil = ToPILImage()

            # Find next largest crop size that is divisible by four.
            crop_size = [int(np.ceil(s/4)*4) for s in img.size]
            lr_crop_size = [s//4 for s in (crop_size)]
            blur_strength = 0  # TODO.

            center_crop = CenterCrop(crop_size[::-1]) # pads s.t. img is div. by 4
            back_crop = CenterCrop(img.size[::-1]) # crop back to orig. img. size

            hr4_gt = img
            hr4_blurred = hr4_gt.filter(ImageFilter.GaussianBlur(blur_strength))
            lr = hr4_blurred.resize(lr_crop_size, Image.BICUBIC)

            elapsed_time, *sr_out = upsample_tensor(model, lr, to_normalize=True, return_time=True)
            hr2_sr, hr4_sr = [back_crop(to_pil(out)) for out in sr_out]
            hr2_bic, hr4_bic = [back_crop(out) for out in upsample_bic(lr)]

            psnr_sr = psnr(hr4_gt, hr4_sr)
            psnr_bic = psnr(hr4_gt, hr4_bic)

            frangi_hr, frangi_sr, frangi_bic = [vessels(i, mask=mask).convert('YCbCr').split()[0] for i in [hr4_gt, hr4_sr, hr4_bic]]

            frangi_acc_sr, frangi_acc_bic = [acc(frangi_hr, other) for other in [frangi_sr, frangi_bic]]
            segmentation_acc_hr, segmentation_acc_sr, segmentation_acc_bic =  [acc(gt, other) for other in [frangi_hr, frangi_sr, frangi_bic]]


            # TODO: Maybe add accuracy for completely black image as well, or use better measure!
            rows += [[model_name, elapsed_time, psnr_sr, psnr_bic, frangi_acc_sr, frangi_acc_bic, segmentation_acc_hr, segmentation_acc_sr, segmentation_acc_bic]]

    columns = ['model_name', 'upscale_time', 'psnr_sr', 'psnr_bic', 'frangi_acc_sr', 'frangi_acc_bic', 'segmentation_acc_hr', 'segmentation_acc_sr', 'segmentation_acc_bic']
    df = pd.DataFrame(data=rows, columns=columns)
    df.to_csv("results.csv", index=None)
 

if __name__ == '__main__':
    main()
