from PIL import Image
from skimage.filters import frangi
from skimage.filters import threshold_otsu
import numpy as np

img = Image.open('../../data/processed/messidor/20051020_43832_0100_PP.tif')
#img = Image.open('../../data/processed/messidor/20051020_57967_0100_PP.tif')
img = np.array(img.convert('YCbCr').split()[0]).astype(float)*255.0

thresh = threshold_otsu(img)
black = img < thresh
vessels = frangi(img, scale_range=(7, 10), beta1=0.5, beta2=600,  black_ridges=True)
vessels[black] = 0.0
#vessels = (vessels > 0.8)*1.0
vessels = Image.fromarray(vessels*255).convert('RGB')
vessels.save('vessels.jpg')
