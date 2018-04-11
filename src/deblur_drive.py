import os
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, CenterCrop, Normalize
from PIL import Image
from PIL.ImageFilter import GaussianBlur

from validate import upsample_tensor, upsample_bic, pad, unpad
from models.lap_srn import LapSRN
from models.unet import UNet

def deblur_tensor(model, lr, return_time=True, to_normalize=False):      
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    normalize = Normalize(mean=mean, std=std)
    to_tensor = ToTensor()
    
    # First create appropiate input image
    lr = to_tensor(lr)
    if to_normalize:
        lr = normalize(lr)
        
    lr = Variable(lr, volatile=True).cuda().unsqueeze(0)     
    def denormalize(out):
        # Undo input[channel] = (input[channel] - mean[channel]) / std[channel]
        for t, m, s in zip(out, mean, std):
            t.mul_(s)
            t.add_(m)
        return out.clamp(0,1)
    
    deblurred = model(lr)
    
    deblurred = denormalize(deblurred[-1].squeeze(0).cpu().data)
    
    elapsed_time = 0 # not used

    if return_time:
        return elapsed_time, deblurred
    else:
        return deblurred
def main():
    drive_dir = Path('../data/raw/DRIVE/test/images/') 
    result_dir = Path('../results/')
    checkpoint_dir = Path('../best_checkpoints/deblur/')

    model_to_checkpoint = lambda m: checkpoint_dir / '{}.pt'.format(m) 

    checkpoint_name = 'unet'

    checkpoint = torch.load(model_to_checkpoint(checkpoint_name))
    if checkpoint_name != 'unet':
        model = LapSRN(depth=5, upsample=False).cuda().eval()
    else:
        model = UNet(3).cuda().eval()
    model.load_state_dict(checkpoint['model_state'])

    blurs = [1,2,3]
    for blur in blurs:
        output_dir_sr = result_dir / 'deblurred_{}_{}'.format(checkpoint_name, blur)
        if not os.path.exists(output_dir_sr):
            os.makedirs(output_dir_sr)

        output_dir_blurred = result_dir / 'blurred_{}'.format(blur)
        if not os.path.exists(output_dir_blurred):
            os.makedirs(output_dir_blurred)

        for file_name in os.listdir(drive_dir):
                print("Deblurring image {}.".format(file_name))

                # Load one drive image
                img = Image.open(drive_dir / file_name)

                to_pil = ToPILImage()

                # UNet expects square images that are div. by 16
                new_size = [max(np.ceil(img.size[0]/16)*16,
                                np.ceil(img.size[1]/16)*16)]*2
                gt, border = pad(img, new_size=new_size)

                lr = gt.filter(GaussianBlur(blur))

                elapsed_time, *deblur_out = deblur_tensor(model, lr, to_normalize=True, return_time=True)

                # Use last output
                deblur_out = unpad(to_pil(deblur_out[-1]), border, cut_off_stripe=0)
                deblur_out.save(output_dir_sr / file_name)

                # Also save lr image.
                unpad(lr, border, cut_off_stripe=0).save(result_dir / 'blurred_{}'.format(blur) / file_name)

if __name__ == '__main__':
    main()
