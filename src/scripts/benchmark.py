import time

import sys
sys.path.append('..')

import numpy as np
import pandas as pd

from torch.autograd import Variable
import torch
from torch import nn
from torch import autograd

from models.lap_srn import LapSRN
from models.unet import UNet

def time_model_single(model, img_size, n_times, num_output):
    times = []
    for i in range(n_times+2):
        img = get_img(img_size)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        if num_output == 1:
            start_event.record()
            up = model(img, num_output=1)
        else:
            start_event.record()
            up = model(img)
        
        end_event.record()
        end_event.synchronize()        
        
        del up
        
        # Discard first measurement (outlier)
        if i != 0:
            times.append(start_event.elapsed_time(end_event)) # ms
    return np.array(times).mean()

def time_model(model, model_name, sizes=[16,32,64]):
    results = []
    for size in sizes:
        num_outputs = [1,2] if model_name != 'unet' else [1]
        for num_output in num_outputs:
            print(model_name, size, num_output)
            result = time_model_single(model, img_size=size, n_times=10, num_output=num_output)
            results.append((model_name, size, num_output, result))
    return results

def get_img(img_size):
    img = Variable(nn.init.constant(torch.Tensor(3,img_size,img_size), val=0.5 ), volatile=True).unsqueeze(0).cuda()
    return img

def main():
    deblur_range = [128, 256, 512, 1024, 1536, 2048]
    upscale_range = [128, 256, 512, 768]

    unet = UNet(num_classes=3).cuda().eval()
    unet_times = time_model(unet, model_name='unet', sizes=deblur_range)
    del unet
    
    lap_deblur = LapSRN(depth=5, upsample=False).cuda().eval()
    lap_deblur_times = time_model(lap_deblur, model_name='lap_deblur', sizes=deblur_range)
    del lap_deblur
    
    lap_srn = LapSRN(depth=10, upsample=True).cuda().eval()
    lap_srn_times = time_model(lap_srn, model_name='lap_srn', sizes=upscale_range)
    del lap_srn

    df = pd.DataFrame(np.vstack((unet_times, lap_deblur_times, lap_srn_times)))
    df.columns = ['model', 'img_size', 'num_outputs', 'time']

    df.to_csv('times.csv', index=None)


if __name__ == '__main__':
    main()
