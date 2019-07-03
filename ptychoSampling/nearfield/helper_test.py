#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import scipy
import skimage, skimage.transform, skimage.data
from skimage.feature import register_translation
from typing import NamedTuple, Optional, List, Tuple
import tensorflow as tf
import attr
from tqdm import tqdm_notebook as tqdm
from pandas import DataFrame



def getSampleObj(npix: int = 256, 
                 mod_range: float = 1, 
                 phase_range: float = np.pi)-> np.ndarray:
    """Creates a sample complex-valued object using stock data from the skimage library.
    
    Uses the stock camera image for the phase and the 
    stock immunohistochemistry image (channel 0) for the modulus [1].
    
    Parameters
    ----------
    npix : 
        Number of pixels in each axis of the object
    
    mod_range : 
        Maximum value of the modulus for the object pixels.
    
    phase_range : 
        Maximum value of the phase for the object pixels.
    
    Returns
    -------
    out : complex ndarray
        A 2d array of shape npix x npix and dtype complex64.
    
    References
    ----------
    .. [1] https://scikit-image.org/docs/dev/api/skimage.data.html
    
    """
    
    phase_img = skimage.img_as_float(skimage.data.camera())[::-1,::-1]
    mod_img = skimage.img_as_float(skimage.data.immunohistochemistry()[:,:,0])[::-1,::-1]
    mod = skimage.transform.resize(mod_img, [npix, npix], 
                                   mode='wrap', preserve_range=True) 
    phase = skimage.transform.resize(phase_img, [npix, npix],
                                     mode='wrap', preserve_range=True)
    
    # Setting the ranges
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase)) * phase_range
    mod = (mod - np.min(mod)) / (np.max(mod) - np.min(mod)) * mod_range
    
    # Centering the phase at 0.
    phase = np.angle(np.exp(1j * (phase - scipy.stats.circmean(phase))))
    
    obj = mod * np.exp(1j * phase)
    return obj.astype('complex64')





