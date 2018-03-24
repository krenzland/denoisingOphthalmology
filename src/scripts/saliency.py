#!/usr/bin/env python3
from PIL import Image


import sys
import numpy as np
import numba
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage import color
from pathlib import Path

def extend_mask(mask, extend_by=1):
    for _ in range(extend_by):
        # Gradient is true at edge of mask
        m_dx, m_dy = [deriv.astype(bool) for deriv in np.gradient(mask*1.0)]
        mask = np.bitwise_or(mask, np.bitwise_or(m_dx, m_dy))
    return mask

def compute_curvature(arr, deriv='gaussian'):
    # http://www0.cs.ucl.ac.uk/staff/S.Arridge/teaching/ndsp/imcurvature.m
    # Note: Individual derivatives might be exchanged, doesn't matter for curvature though!
    deriv = 'gaussian'
    if deriv == 'gaussian':
        sigma=1
        img_dx = ndimage.gaussian_filter(arr, order=[1,0], sigma=sigma)
        img_dy = ndimage.gaussian_filter(arr, order=[0,1], sigma=sigma)
        img_dxx = ndimage.gaussian_filter(arr, order=[2,0], sigma=sigma)
        img_dyy = ndimage.gaussian_filter(arr, order=[0,2], sigma=sigma)
        img_dxy = ndimage.gaussian_filter(arr, order=[1,1], sigma=sigma)
        img_dyx = img_dxy
    else:
        sobel = 1/8 * np.array([-1,0,1,-2,0,2,-1,0,1]).reshape(3,3)
        img_dx = ndimage.convolve(arr, sobel)
        img_dy = ndimage.convolve(arr, sobel.T)
        img_dxx = ndimage.convolve(img_dx, sobel)
        img_dyy = ndimage.convolve(img_dy, sobel.T)
        img_dxy = ndimage.convolve(img_dx, sobel.T)
        img_dyx = ndimage.convolve(img_dy, sobel)

    gradient_magnitude = np.sqrt(img_dx**2 + img_dy**2)
    curvature = 1.0/gradient_magnitude

    invalid = (arr <= 0.0) | (gradient_magnitude == 0.0)
    curvature[invalid] = 0.0

    # mask is true where grad is possibly zero
    mask = np.zeros_like(invalid, dtype=bool)
    mask[invalid] = True

    #mask1 = extend_mask(mask)
    mask2 = extend_mask(mask, extend_by=2)

    curvature = img_dy**2 * img_dxx - img_dx * (img_dxy + img_dyx) * img_dy + img_dx**2 * img_dyy
    curvature[~mask2] = curvature[~mask2] / gradient_magnitude[~mask2]**3
    curvature[mask2] = 0.0
    
    return curvature.clip(0,1)

@numba.stencil(neighborhood=((-3, 3),(-3,3)), cval=-42.0)
def entropy_kernel(hist):
    # First compute local probabilities (disc. using passed bins)
    count = np.zeros(8)
    
    it = np.array([-3,-2,-1,0, 1,2,3])
    for i in it:
        for j in it:
            count[hist[i,j]] += 1

    probs = count / count.sum() # normalize
    
    # Now compute entropy.
    entropy = 0.0
    for i in it:
        for j in it:      
            p = probs[hist[i,j]]
            log_p = np.log(p)
            entropy += -1 * p * log_p    
    return entropy

def compute_entropy(arr):
    edges = np.array([  0.   ,  31.875,  63.75 ,  95.625, 127.5  , 159.375, 191.25 ,
        223.125, 255.   ])/255.0

    # Reflect pad for better stability at borders
    pad_width = 3
    arr = np.pad(arr, pad_width=pad_width, mode='reflect')
    
    # Discretise intensity using edges.
    print(0, len(edges)-2)
    hist = np.digitize(arr, edges, right=True).clip(0, len(edges)-2) # interval should be open on both ends
    entr = entropy_kernel(hist)
    
    # Remove padding
    entr = entr[pad_width:-pad_width, pad_width:-pad_width]
    print(entr.min(), entr.max())
    
    # Finally use a Gaussian low pass filter to remove small elements
    entr = ndimage.gaussian_filter(entr, 0.5, truncate=3)

    # Normalize to (0,1)
    entr = entr/entr.max()
    
    # Make sure the invalid constant padding is removed!
    assert(entr.min() >= 0.0)
    
    return entr

@numba.jit(nopython=True, parallel=True)
def get_neighbour_distance(size_sqrt=7):
    assert(size_sqrt % 2 == 1)
    size = size_sqrt**2
    d = np.zeros(size).reshape(size_sqrt, size_sqrt)
    center = size_sqrt//2
    for i in range(size_sqrt//2+1):
        for j in range(size_sqrt//2+1):
            d[center+i][center+j] = (i**2 + j**2)**0.5
            d[center+i][center-j] = (i**2 + j**2)**0.5
            d[center-i][center+j] = (i**2 + j**2)**0.5
            d[center-i][center-j] = (i**2 + j**2)**0.5
    d = np.exp(-d)
    return d

@numba.stencil(neighborhood=((-3, 3),(-3,3)), standard_indexing=('neighborhood',), cval=-42.0)
def uniq_kernel(feature_map, neighborhood):
    feature_center = feature_map[0,0]
    uniqueness = 0.0
    for i in range(7):
        for j in range(7):
            feature_dist = np.abs(feature_center - feature_map[-3+i,-3+j])
            uniqueness += neighborhood[i,j] * feature_dist
    return uniqueness
    
def uniqueness(feature_map):
    # Reflect pad for better stability at borders
    pad_width = 3
    feature_map = np.pad(feature_map, pad_width=pad_width, mode='reflect')

    neighborhood = get_neighbour_distance(size_sqrt=7)
    uniq = uniq_kernel(feature_map, neighborhood)
    
    uniq = uniq[pad_width:-pad_width, pad_width:-pad_width]

    # Make sure the invalid constant padding is removed!
    assert(uniq.min() >= 0.0)

    return uniq/uniq.max()

def saliency(img):
    arr = color.rgb2gray(np.array(img))
    arr = np.float32(arr)

    # Ignore background
    black = threshold_otsu(arr)
    black = arr < black

    curvature = compute_curvature(arr)
    entropy = compute_entropy(arr)
    uniq_curv = uniqueness(curvature) 
    uniq_entr =  uniqueness(1.0-entropy)
    
    saliency = 0.4 * uniq_curv + 0.6 * uniq_entr
    saliency[black] = 0.0
    saliency = Image.fromarray(saliency*255.0).convert('RGB')
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
