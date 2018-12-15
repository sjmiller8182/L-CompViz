"""imtools.py

    Tools for working with images


    get_imlist - gets pathes to images for a path
    im_resize - resize an image (as np.array)
    equalize_hist - equalize histogram of image
    im_average - average a set of images
"""

import glob
import os
from PIL import Image
import numpy as np

def get_imlist(path, extension = '.jpg'):
    """
    Returns a list of images at the specified path with the specified extension
    
    path: string
        image search path
    extension: string
        extension of image to glob for
    
    Return: dict
        a dict of pathes keyed by image names (with extension)
    
    """
    
    ims = list()
    d_ims = dict()
    
    ims = glob.glob(path + '*' + extension)
    for im in ims:
        d_ims[im.split('\\')[-1]] = im
    return d_ims
    
def im_resize(im, size):
    """
    Resizes an image array using PIL
    
    im: np.array
        image array
    size: tuple
        new size for image
    
    Return: array
        np.array of image
    
    """
    
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize(size))
    
def equalize_hist(im, num_bins = 256):
    """
    Histogram equalization for greyscale images
    
    im: np.array
        image array
    num_bins: number of bins to use
        new size for image
    
    Return: list
        [0]: np.array of image
        [1]: cdf of image    
    
    """
    
    imhist, bins = np.histogram(im.flatten(), num_bins, normed = True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf
    
def im_average(im_paths, resize = None):
    """
    Compute averages of images in the list (greyscaled)
    
    im: list
        a list of image paths
    resize: tuple
        resize images to a standard size
    
    Return: np.array
        an average of images as an array
    
    """
    
    
    if resize is not None:
        average_im = Image.open(im_paths[0]).resize(resize)
    else:
        average_im = Image.open(im_paths[0])
    
    # seed average placeholder
    average_im = np.array(average_im, 'f')
    
    for path in im_paths[1:]:
        try:
            im = Image.open(path)
        except:
            print("unable to open " + str(path))
            
        if resize is not None:
            im = im.resize(resize)
            
        average_im += np.array(im, 'f')
    
    # calc average
    average_im /= len(im_paths)
    
    return np.array(average_im, 'uint8')