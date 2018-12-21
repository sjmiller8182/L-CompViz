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
    
def _plot_harris_points(image, filtered_corrds, figsize = (10,7), cmap = 'gray'):
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
    plt.imshow(image, cmap = cmap)
    plt.plot([p[1] for p in filtered_corrds], [p[0] for p in filtered_corrds], '*')

def _find_corresponding_points(twosided = True):
    # TODO
    pass

def _get_image_descriptors(image, filtered_corrds, width = 5):
    '''For each point in the returned corrds, generate patchs around the points.
    '''
    
    desc = list()
    for corrd in filtered_corrds:
        patch = image[corrd[0] - width:corrd[0] + width + 1,
                     corrd[1] - width:corrd[1] + width + 1].flatten()
        desc.append(patch)
    return desc

def _match_descriptors(desc1, desc2, threshold = 0.5):
    '''Match the first descriptor to the second descriptor using normalized corss-correlation.
    '''
    
    n = len(desc1[0])
    
    d = -np.ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
        for j in range(len(desc2)):
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_val = np.sum(d1 * d2) / (n-1)
            if ncc_val > threshold:
                d[i,j] = ncc_val
    ndx = np.argsort(-d)
    matchscores = ndx[:,0]
    return matchscores    
        
def _match_descriptors_twosided(desc1, desc2, threshold = 0.5):
    '''
    '''
    
    # run matches from both images
    match12 = _match_descriptors(desc1, desc2, threshold)
    match21 = _match_descriptors(desc2, desc1, threshold)
    
    ndx12 = np.where(matches12 >= 0)[0]
    
    # remove unsymmetric matches
    for n in ndx12:
        if matches21[matches12[n]] != n:
            matches12[n] = -1
            
    return matches12
        
class Harris:
    '''Harris corner detector
    '''
    
    def __init__(self):
        self.im = None
        self.fit_points = None
        self.descriptors = None
        self.matching_points = None
        
    def fit(self, im, sigma = 3, min_dist = 10, threshold = 0.1, desc_width = 5):
        '''Run Harris detector on images in list
        '''
        self.im = im
        harris_im = _compute_harris_response(im, sigma)
        self.fit_points = _get_harris_points(harris_im, min_dist, threshold)
        self.descriptors = _get_image_descriptors(im, self.fit_points, width = desc_width)
        
    def plot(self, figsize = (10,7)):
        '''
        '''
        _plot_harris_points(self.im, self.fit_points, figsize, cmap = 'gray')
        
    def get_points(self):
        '''Return the Harris points
        '''
        return self.fit_points
        
    def get_im()
        '''Return the images used
        '''
        return self.im
    
    def match_corresponding_points(self, ims, threshold, twosided = False):
        '''Match interest points in target image to interest points in other images
        
        ims: list of np.array
            images to compare as np.array; input as list
        threshold: float
            cross-correlation threshold
        twosided: bool
            True: compare images twosided, False (default): compare from target image only
        
        return: list
            matching corresponding points based on threshold
        '''
        
        