import os
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import ToTensor, ToPILImage, CenterCrop
from PIL import Image

from validate import upsample_tensor, upsample_bic
from model import LapSRN

def main():
    drive_dir = Path('../data/raw/DRIVE/test/images/') 
    result_dir = Path('../results/')
    checkpoint_dir = Path('../best_checkpoints/')

    model_to_checkpoint = lambda m: checkpoint_dir / '{}.pt'.format(m) 

    checkpoint = torch.load(model_to_checkpoint('gan'))
    model = LapSRN(depth=10).cuda().eval()
    model.load_state_dict(checkpoint['model_state'])

    for file_name in os.listdir(drive_dir):
            print("Upscaling image {}.".format(file_name))

            # Load one drive image
            img = Image.open(drive_dir / file_name)

            to_pil = ToPILImage()

            # Find next largest crop size that is divisible by four.
            crop_size = [int(np.ceil(s/4)*4) for s in img.size]
            lr_crop_size = [s//4 for s in (crop_size)]
            blur_strength = 0.0  # TODO.

            center_crop = CenterCrop(crop_size[::-1]) # pads s.t. img is div. by 4
            back_crop = CenterCrop(img.size[::-1]) # crop back to orig. img. size

            hr4_gt = img
            lr = hr4_gt.resize(lr_crop_size, Image.BICUBIC)

            elapsed_time, *sr_out = upsample_tensor(model, lr, to_normalize=True, return_time=True)
            print("Elapsed time: {}[ms]".format(elapsed_time*1000))

            # Save largest upscaled image.
            out_sr = back_crop(to_pil(sr_out[-1]))
            out_sr.save(result_dir / 'sr' / file_name)

            # Also save bicubic image.
            bic_out = upsample_bic(lr)
            bic_out = back_crop(bic_out[-1])
            bic_out.save(result_dir / 'bic' / file_name)

if __name__ == '__main__':
    main()
