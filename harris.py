"""harris.py

    Implementation of Harris Corner Detection

    function - what it does
    
"""

import numpy as np
from scipy.ndimage import filters
from PIL import Image
import matplotlib.pyplot as plt

def _compute_harris_response(im, sigma = 3):
    '''Compute the Harris corener detector response function for each pixel in a greyscale image.
    
    im: np.array
        greyscaled image as an array
    sigma: float 
        gaussian filter scale
    
    Return: float
        Harris indicator ratio
    '''
    
    # get gradients
    imx = np.zeros(im.shape)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0,1), imx)
    filters.gaussian_filter(im, (sigma, sigma), (1,0), imy)
    
    # computer components of the Harris Matrix
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)
    
    # calc determinate and trace
    Wdet = Wxx*Wyy-Wxy**2
    Wtr = Wxx + Wyy
    
    return Wdet / Wtr
    
def _get_harris_points(harris_im, min_dist = 10, threshold = 0.1):
    '''Returns corner from a Harris response image.
    
    harris_im: np.array
        image of harris responses
    min_dist: int
        minimum number of pixels separating the corners and the image boundry
    threshold: 
        response threshold
    
    Return: np.array
        corrds of interesting points
    '''
    
    # find best candidate pixels
    corner_threshold = harris_im.max() * threshold
    harris_im_t = (harris_im > corner_threshold) * 1
    
    # get coordinate of the candidates
    corrds = np.array(harris_im_t.nonzero()).T
    
    # get candidate values
    candidate_values = [harris_im[c[0],c[1]] for c in corrds]
    index = np.argsort(candidate_values)
    
    # filter based on min dist in the image boundry
    allowed_locations = np.zeros(harris_im.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    
    # select the best points accounting for min_dist between pixels
    filtered_corrds = list()
    for i in index:
        if allowed_locations[corrds[i,0],corrds[i,1]] == 1:
            filtered_corrds.append(corrds[i])
            allowed_locations[(corrds[i,0]-min_dist):(corrds[i,1])+min_dist,
                             (corrds[i,1]-min_dist):(corrds[i,1])+min_dist] = 0
    
    return filtered_corrds
    
def _plot_harris_points(image, filtered_corrds, figsize = (10,7)):
    '''Plots Harris points on the image
    
    image: np.array
        image Harris points were calculated on
    filtered_corrds: np.array
        Harris points
    figsize: tuple
        size of the plot figure, default: (10,7)
        
    return: None
    '''
    plt.figure(figsize = figsize)
    plt.axis('off')
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_corrds], [p[0] for p in filtered_corrds], '*')
    
class Harris:
    '''Harris corner detector
    '''
    
    def __init__(self):
        self.im = None
        self.fit_points = None
        
    def fit(self, im, sigma = 3, min_dist = 10, threshold = 0.1):
        self.im = im
        harris_im = _compute_harris_response(im, sigma)
        self.fit_points = _get_harris_points(harris_im, min_dist, threshold)
        
    def plot(self, figsize = (10,7)):
        _plot_harris_points(self.im, self.fit_points, figsize)
        
    def points(self):
        return self.fit_points