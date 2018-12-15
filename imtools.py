"""imtools.py

    Tools for working with images


    get_imlist - gets pathes to images for a path
    im_resize - resize an image (as np.array)
    equalize_hist - equalize histogram of image
    im_average - average a set of images
    denoise - removes noise from and image
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
    
def pca(X):
    """
    Compute averages of images in the list (greyscaled)
    
    im: list
        a list of image paths
    resize: tuple
        resize images to a standard size
    
    Return: np.array
        an average of images as an array
    
    """
    
    # get dimensions
    num_data, dim = X.shape
    
    # center data
    mean_X = X.mean(axis = 0)
    X = X-mean_X
    
    # PCA
    if dim > num_data:
        M = np.dot(X,X.T)
        e, ev = np.linalg.eigh(M)
        tmp = np.dot(X.T, ev).T
        V = tmp[::-1]
        S = np.sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]
    return V, S, mean_X
    
def denoise(im, U_init, tolerance = 0.1, tau = 0.125, tv_weight = 100):
    
    m, n = im.shape
    
    # init
    U = U_init
    Px = im
    Py = im
    error = 1
    
    while(error > tolerance):
        Uold = U
        
        # calc grads
        GradUx = np.roll(U,-1,axis = 0)
        GradUy = np.roll(U,-1,axis = 1)
        
        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NewNorm = np.maximum(1,np.sqrt(PxNew**2 + PyNew**2))
        
        Px = PxNew/NewNorm
        Py = PyNew/NewNorm
        
        RxPx = np.roll(Px, 1, axis = 1)
        RyPy = np.roll(Py, 1, axis = 0)
        
        DivP = (Px-RxPx)+(Py-RyPy)
        
        U = im + tv_weight*DivP
        
        error = np.linalg.norm(U-Uold)/np.sqrt(m*n)
        
    return U, im-U