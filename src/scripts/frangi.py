from PIL import Image
from skimage.filters import frangi
from skimage.filters import threshold_otsu
import sys
import numpy as np
from pathlib import Path

def vessels(img):
    img = np.array(img.convert('YCbCr').split()[0]).astype(float)

    # Ignore background
    thresh = threshold_otsu(img)
    black = img < thresh

    # Compute vessels
    vessels = frangi(img, scale_range=(7, 10), beta1=0.5, beta2=6,  black_ridges=True)

    # Reset black pixels to black. Otherwise we get a surrounding ring.
    vessels[black] = 0.0
    vessels = Image.fromarray(vessels*255).convert('RGB')
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
