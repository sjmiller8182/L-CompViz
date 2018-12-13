"""imtools.py

    Tools for working with images


    get_imlist - gets pathes to images for a path
"""

import glob
import os

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