#!/usr/bin/env python3
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Normalize, CenterCrop
from PIL import Image, ImageFilter, ImageOps
import PIL.ImageChops as chop
from skimage.filters import frangi, threshold_otsu, sobel
from skimage import measure
from skimage.io._plugins.pil_plugin import pil_to_ndarray, ndarray_to_pil

#from dataset import Dataset, Split, SplitDataset, get_lr_transform, get_hr_transform
from models.lap_srn import LapSRN

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

def acc(a, b, transform=lambda x:x):
    a = a.convert('YCbCr').split()[0]
    b = a.convert('YCbCr').split()[0]
    a = transform(a)
    b = transform(b)
    a = (np.array(a)/255.0).astype(bool)
    b = (np.array(b)/255.0).astype(bool)
    assert(a.size == b.size)
    size = a.size
    correct = (a == b).sum()
    return correct/size * 100.0

def acc(y, y_hat, transform=lambda x:x):
    y = np.array(transform(y))
    y_hat = np.array(transform(y_hat))
    total = y.size * 1.0
    correct = ((y == y_hat)*1.0).sum()

    return (correct/total)*100

def mse(y, y_hat, transform=lambda x:x):
    y = np.array(transform(y))
    y_hat = np.array(transform(y_hat))
    
    diff = y_hat/255.0 - y/255.0
    return np.sum(diff**2)/diff.size    

def psnr(y, y_hat, transform=lambda x:x):
    error = mse(y, y_hat, transform)
    psnr = -10 * np.log10(error)
    return psnr

def edges(img):
    arr = pil_to_ndarray(img.convert('YCbCr').split()[0])
    return ndarray_to_pil(sobel(arr))

def vessels(img, mask):
    params = {'beta1': 0.7,
              'beta2': 0.01,
              'scale_max': 3,
              'scale_min': 0,
              'threshold': 0.2} 

    img = np.array(img.convert('YCbCr').split()[0]).astype(float)/255.

    # Ignore background
    thresh = threshold_otsu(img, nbins=255)
    black = img < thresh

    # Compute vessels
    vessels = frangi(img, scale_range=(params['scale_min'], params['scale_max']),
                     beta1=params['beta1'],
                     beta2=params['beta2'],
                     black_ridges=True)

    # Reset black pixels to black. Otherwise we get a surrounding ring.
    vessels[black] = 0.0
    vessels = (vessels > params['threshold'])
    vessels = Image.fromarray(vessels*255.0)
    return vessels

def pad(img, new_size=None, color='black'):
    img = np.array(img).transpose(2,0,1) # Channel first
    size = np.array(img.shape)[1:]
    if new_size is None:
        new_size = np.ceil(size/4) * 4
    diff = new_size - size
    border = np.zeros(4, dtype=np.int)
    border[0] = diff[0]//2
    border[1] = diff[1]//2
    border[2] = diff[0] - border[0]
    border[3] = diff[1] - border[1]

    new_img = np.zeros((3, int(new_size[0]),
                        int(new_size[1])))
    for i,channel in enumerate(img):
        channel_pad = np.pad(channel,
                     ((border[0], border[2]), 
                      (border[1], border[3])),
                     'reflect')
        new_img[i,:,:] = channel_pad
    new_img = np.uint8(new_img).transpose(1,2,0)

    new_img = Image.fromarray(new_img)
    return new_img, border

def unpad(img, border, cut_off_stripe=4):
    img = np.array(img)
    new_size = np.array(img.shape[0:-1])
    # Remove additional stripe of 4 px
    border = border + cut_off_stripe
    if len(img.shape) == 3:
        img = img[border[0]:img.shape[0]-border[2],
                border[1]:img.shape[1]-border[3],
                :]
    else:
        # only 1 channel
        img = img[border[0]:img.shape[0]-border[2],	
                border[1]:img.shape[1]-border[3]]
 
    img = Image.fromarray(img)
    return img

def main():
    checkpoint_dir = Path('../best_checkpoints/')
    models = ['gan_notfinal', 'saly_perc', 'saliency', 'perceptual']
    model_to_checkpoint = lambda m: checkpoint_dir / '{}.pt'.format(m) 

    rows = []
    for model_name in models:
        checkpoint = torch.load(model_to_checkpoint(model_name))

        model = LapSRN(depth=10).cuda().eval()
        model.load_state_dict(checkpoint['model_state'])

        drive_dir = Path('../data/raw/DRIVE/test/') 
        drive_range = range(1,20+1)

        for num in drive_range:
            num = str(num).zfill(2)
            print("Analysing image {} for model {}.".format(num, model_name))
            # Load one drive image with mask and ground truth segmentation.
            mask = Image.open(drive_dir / 'mask/{}_test_mask.gif'.format(num))
            img = Image.open(drive_dir / 'images/{}_test.tif'.format(num))
            gt = Image.open(drive_dir / '1st_manual/{}_manual1.gif'.format(num)).convert('YCbCr').split()[0]

            to_pil = ToPILImage()

            # Find next largest crop size that is divisible by four.
            hr4_gt, border = pad(img)

            lr_crop_size = [s//4 for s in hr4_gt.size]
            lr = hr4_gt.resize(lr_crop_size, Image.BICUBIC)

            elapsed_time, *sr_out = upsample_tensor(model, lr, to_normalize=True, return_time=True)
            hr2_sr, hr4_sr = [unpad(to_pil(out), border) for out in sr_out]
            hr2_bic, hr4_bic = [unpad(out, border) for out in upsample_bic(lr)]

            hr4_gt = unpad(hr4_gt, border)
            psnr_sr = measure.compare_psnr(np.array(hr4_gt), np.array(hr4_sr))
            psnr_bic = measure.compare_psnr(np.array(hr4_gt), np.array(hr4_bic))

            ssim_sr = measure.compare_ssim(np.array(hr4_gt), np.array(hr4_sr), data_range=256, multichannel=True)
            ssim_bic = measure.compare_ssim(np.array(hr4_gt), np.array(hr4_bic), data_range=256, multichannel=True)

            sobel_sr, sobel_bic = [mse(hr4_gt, out, edges) * 10e4 for out in [hr4_sr, hr4_bic]]

            frangi_hr, frangi_sr, frangi_bic = [vessels(i, mask=mask).convert('YCbCr').split()[0] for i in [hr4_gt, hr4_sr, hr4_bic]]

            frangi_acc_sr, frangi_acc_bic = [acc(frangi_hr, other) for other in [frangi_sr, frangi_bic]]
            segmentation_acc_hr, segmentation_acc_sr, segmentation_acc_bic =  [acc(unpad(gt, border=np.zeros(4, dtype=int)), other) for other in [frangi_hr, frangi_sr, frangi_bic]]


            # TODO: Maybe add accuracy for completely black image as well, or use better measure!
            rows += [[model_name, elapsed_time, psnr_sr, psnr_bic, ssim_sr, ssim_bic, sobel_sr, sobel_bic, \
                      frangi_acc_sr, frangi_acc_bic, segmentation_acc_hr, segmentation_acc_sr, segmentation_acc_bic]]

    columns = ['model_name', 'upscale_time', 'psnr_sr', 'psnr_bic', 'ssim_sr', 'ssim_bic', 'sobel_sr', 'sobel_bic', \
               'frangi_acc_sr', 'frangi_acc_bic', 'segmentation_acc_hr', 'segmentation_acc_sr', 'segmentation_acc_bic']
    df = pd.DataFrame(data=rows, columns=columns)
    df.to_csv("results.csv", index=None)
 

if __name__ == '__main__':
    main()
