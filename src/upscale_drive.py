import os
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import ToTensor, ToPILImage, CenterCrop
from PIL import Image

from validate import upsample_tensor, upsample_bic, pad, unpad
from models.lap_srn import LapSRN

def main():
    drive_dir = Path('../data/raw/DRIVE/test/images/') 
    result_dir = Path('../results/')
    checkpoint_dir = Path('../best_checkpoints/')

    model_to_checkpoint = lambda m: checkpoint_dir / '{}.pt'.format(m) 

    checkpoint_name = 'perceptual'
    checkpoint = torch.load(model_to_checkpoint(checkpoint_name))
    model = LapSRN(depth=10).cuda().eval()
    model.load_state_dict(checkpoint['model_state'])

    output_dir_sr = result_dir / 'sr_{}'.format(checkpoint_name)
    if not os.path.exists(output_dir_sr):
        os.makedirs(output_dir_sr)

    for file_name in os.listdir(drive_dir):
            print("Upscaling image {}.".format(file_name))

            # Load one drive image
            img = Image.open(drive_dir / file_name)

            to_pil = ToPILImage()

            # Find next largest crop size that is divisible by four.
            blur_strength = 0.0  # TODO.


            hr4_gt, border = pad(img)
            lr_crop_size = [s//4 for s in (hr4_gt.size)]

            lr = hr4_gt.resize(lr_crop_size, Image.BICUBIC)

            elapsed_time, *sr_out = upsample_tensor(model, lr, to_normalize=True, return_time=True)
            print("Elapsed time: {}[ms]".format(elapsed_time*1000))

            # Save largest upscaled image.
            out_sr = unpad(to_pil(sr_out[-1]), border, cut_off_stripe=0)
            out_sr.save(output_dir_sr / file_name)

            # Also save bicubic image.
            bic_out = upsample_bic(lr)
            bic_out = unpad(bic_out[-1], border, cut_off_stripe=0)
            bic_out.save(result_dir / 'bic' / file_name)

if __name__ == '__main__':
    main()
