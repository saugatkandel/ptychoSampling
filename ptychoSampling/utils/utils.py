import numpy as np
from typing import Tuple, Union
import scipy
import skimage

def generalized2dGaussian(xdata_tuple: np.ndarray, # array consisting of 2d points.
                          amplitude: float,
                          center_x: float,
                          center_y: float,
                          sigma_x: float,
                          sigma_y: float,
                          theta: float,
                          offset: float) -> np.ndarray:

    XX = xdata_tuple[:,0]
    YY = xdata_tuple[:,1]

    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(-(a * ((XX - center_x) ** 2)
                                      + 2 * b * (XX - center_x) * (YY -center_y)
                                      + c * ((YY - center_y) ** 2)))
    return g


def getSpeckle(shape: Tuple[int, int],
               window_size: int) -> np.ndarray:
    r"""Generates a speckle pattern.

    To generate a speckle pattern, this function uses a ``window_size x window_size`` array of complex numbers
    with unit amplitude and uniformly random phase. This array is padded with zeros to get an ``npix x npix`` array,
    an FFT of which gives us a speckle pattern. The speckle pattern thus generated is discontinuous;
    there is a phase step of pi between adjacent pixels in both the ``x`` and ``y`` directions.
    We remove these discontinuities to get the final, continuous, speckle pattern.

    Parameters
    ----------
    shape : tuple(int, int)
        Number of pixels along each side of the 2d array containing the speckle pattern.
    window_size : int
        The size of the rectangular window used to generate the speckle pattern.
        Larger window sizes give smaller speckle sizes and vice versa.
        (*Note*: I tried a circular window as well, but the results did not change noticeably.)

    Returns
    --------
    out : ndarray(complex)
        A 2d array of size ``npix x npix`` and dtype complex64.
    """
    nx = shape[1]
    ny = shape[0]
    if (window_size > nx) or (window_size > ny) :
        raise ValueError("Window size should be smaller than the size of output array.")

    # generating the random array
    ran = np.exp(1j * np.random.rand(ny, nx) * 2 * np.pi)

    window = np.zeros((ny, nx))
    indx1 = ny // 2 - window_size // 2
    indx2 = ny // 2 + window_size // 2
    window[indx1: indx2, indx1: indx2] = 1

    # Circular window - doesn't change the results.
    # xx, yy = np.meshgrid(np.arange(npix), np.arange(npix))
    # mask = ((xx - npix // 2)**2 + (yy - npix // 2)**2 < (window_size // 2)**2)
    # window[mask] = 1

    t = window * ran

    ft = np.fft.fftshift(np.fft.fft2(t, norm='ortho'))
    absvals = np.abs(ft)
    angvals = np.angle(ft)

    # Removing the discontinuities in the phases
    angvals[::2] = (angvals[::2] + np.pi) % (2 * np.pi)
    angvals[:, ::2] = (angvals[:, ::2] + np.pi) % (2 * np.pi)
    return (absvals * np.exp(1j * angvals)).astype('complex64')


def getRandomComplexArray(shape: Tuple[int, ...],
                          mod_range: float = 1.0,
                          phase_range: float = np.pi) -> np.ndarray:
    rands = np.random.random((2, *shape))
    return rands[0] * mod_range * np.exp(1j * rands[1] * phase_range)

def upSample(array2d: np.ndarray,
             upsampling_factor: int) -> np.ndarray:
    upsampled_npix = np.array(array2d.shape) * upsampling_factor
    upsampled_array = np.repeat(np.repeat(array2d[:, None, :, None], upsampled_npix[0], axis=3),
                                upsampled_npix[1], axis=1)
    upsampled_array = upsampled_array.reshape(upsampled_npix[0], upsampled_npix[1]) / np.product(upsampled_npix)
    return upsampled_array

def downSample(array2d: np.ndarray,
               downsampling_factor: int) -> np.ndarray:
    d = downsampling_factor
    n = array2d.shape // downsampling_factor
    downsampled_array = np.sum(np.reshape(array2d, [n[0], d, n[1], d]), axis=(1,3))
    return downsampled_array
