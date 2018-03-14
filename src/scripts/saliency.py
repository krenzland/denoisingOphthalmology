#!/usr/bin/env python3
from PIL import Image
from skimage import io, color
import skimage.filters as fil

import sys
import numpy as np
from pathlib import Path

def saliency(img):
    """
    Computes the curvedness of the image and use as saliency.
    """
    arr = color.rgb2gray(np.array(img))
    arr = np.float32(arr)

    # Ignore background
    black = fil.threshold_otsu(arr)
    black = arr < black

    # Pad
    pad_width = 40
    arr = np.pad(arr, pad_width=pad_width, mode='reflect')

    # Add slight blur for stability
    arr = fil.gaussian(arr, sigma=1.5)
    
    # Compute image derivatives
    img_dx = fil.scharr_h(arr)
    img_dy = fil.scharr_v(arr)
    img_dxx = fil.scharr_h(img_dx)
    img_dyy = fil.scharr_v(img_dy)
    img_dxy = fil.scharr_v(img_dx)
    
    # Assemble to curvedness
    curvedness = np.sqrt(img_dxx**2 + 2* img_dxy**2 + img_dyy**2) 

    # Unpad.
    c = curvedness[pad_width:-pad_width, pad_width:-pad_width].copy()

    # Remove black border
    c[black] = 0

    # Reweight image, set weight equal for all heavily weighted objects.
    hi = np.percentile(curvedness, q=[97.5])[0]
    c = c.clip(0.0, hi)

    # Finally convert to image.
    c = c/c.max()
    saliency = Image.fromarray(c*255.0).convert('RGB')
    return saliency

def main():
    in_name = Path(sys.argv[1])
    out_name = in_name.parent / 'saliency' / (in_name.stem + '.jpg')

    in_name = str(in_name)
    out_name = str(out_name)

    img = Image.open(in_name)

    out = saliency(img)
    out.save(out_name)
    print('Wrote {}.'.format(out_name))

if __name__ == '__main__':
    main()
