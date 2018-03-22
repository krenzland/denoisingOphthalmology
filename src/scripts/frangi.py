#!/usr/bin/env python3
from PIL import Image
from skimage.filters import frangi
from skimage.filters import threshold_otsu
import sys
import numpy as np
from pathlib import Path

def vessels(img):
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
    vessels = Image.fromarray(vessels*255.0).convert('RGB')
    return vessels

def main():
    in_name = Path(sys.argv[1])
    out_name = in_name.parent / 'vessels' / (in_name.stem + '.jpg')

    in_name = str(in_name)
    out_name = str(out_name)

    img = Image.open(in_name)

    out = vessels(img)
    out.save(out_name)
    print('Wrote {}.'.format(out_name))

if __name__ == '__main__':
    main()
