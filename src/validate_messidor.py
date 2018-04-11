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

from validate import *
from models.lap_srn import LapSRN
from dataset import Dataset, SplitDataset, Split

def main():
    from torchvision.transforms import Normalize
    dataset = Dataset(path='../data/processed/messidor', 
                    hr_transform=None,
                    lr_transform=None,
                    preload=False)
    validation_files = dataset.filenames[int(len(dataset.filenames) * 0.8):]
    print(len(validation_files))
    del dataset
    
    checkpoint_dir = Path('../best_checkpoints/')
    models = ['gan', 'saly_perc', 'saliency', 'perceptual']
    model_to_checkpoint = lambda m: checkpoint_dir / '{}.pt'.format(m) 

    rows = []
    for model_name in models:
        print('Evaluating ', model_name)
        checkpoint = torch.load(model_to_checkpoint(model_name))

        model = LapSRN(depth=10).cuda().eval()
        model.load_state_dict(checkpoint['model_state'])

        for img_path in validation_files:
            print("Analysing image {} for model {}.".format(img_path, model_name))
            img = Image.open(img_path)

            to_pil = ToPILImage()

            # Find next largest crop size that is divisible by four.
            for factor in [2,4]:
                size = np.array(img.size)
                new_size = np.ceil(size/factor) * factor
                gt, border = pad(img, new_size[::-1]) # axis are switched for pad
            
                lr_crop_size = [s//factor for s in gt.size]
                lr = gt.resize(lr_crop_size, Image.BICUBIC)

                sr_out = upsample_tensor(model, lr, to_normalize=True, return_time=False, factor=factor)[-1]
                sr_out = unpad(to_pil(sr_out), border, cut_off_stripe=factor)
                bic_out = unpad(upsample_bic(lr)[factor//2-1], border, cut_off_stripe=factor)
                gt = unpad(gt, border, cut_off_stripe=factor)

                psnr_sr = psnr(gt, sr_out)
                psnr_bic = psnr(gt, bic_out)

                ssim_sr = ssim(gt, sr_out)
                ssim_bic = ssim(gt, bic_out)

                sobel_sr, sobel_bic = [edge_error(gt, out) * 10e4 for out in [sr_out, bic_out]]

                rows += [[model_name, factor, img_path, psnr_sr, psnr_bic, ssim_sr, ssim_bic, sobel_sr, sobel_bic]]

        print('Saving intermed. csv!')
        columns = ['model_name', 'factor', 'img_path', 'psnr_sr', 'psnr_bic', 'ssim_sr', 'ssim_bic', 'sobel_sr', 'sobel_bic']
        df = pd.DataFrame(data=rows, columns=columns)
        df.to_csv("results_messidor.csv", index=None)

if __name__ == '__main__':
    main()
