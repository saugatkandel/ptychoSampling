#Author - Saugat Kandel
# coding: utf-8


"""
Port of matlab code from:
https://github.com/flatironinstitute/NoRMCorre/blob/master/dftregistration_min_max_3d.m

Original 2d version by Manuel Guizar:
https://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation

Adapted from the 2d registration code in (which is based on Manuel Guizar's code):
https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/register_translation.py#L109

by Saugat Kandel - 04/17/18


"""



import numpy as np



def _upsampled_dft(data, upsampled_region_size=None,
                   upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.
    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.
    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.
    Parameters
    ----------
    data : 3D ndarray
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.  If None, this is
        equal to ``data.shape``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)
    Returns
    -------
    output : 3D ndarray
            The upsampled DFT of the specified region.
    """
    
    nr, nc, nz = data.shape
    
    if upsampled_region_size is None:
        upsampled_region_size = data.shape
    # if people pass in an integer, expand it to a list of equal-sized sections
    elif not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")
    upsampled_region_size = np.array(upsampled_region_size, dtype='int')
    upsampr, upsampc, upsampz = upsampled_region_size
    
    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    elif not hasattr(axis_offsets, "__iter__"):
        axis_offsets = [axis_offsets, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")
            
    offsr, offsc, offsz = axis_offsets
    

    #kernc = np.exp((-1j * 2 * np.pi / (nc * upsample_factor)) * 
    #               (np.fft.ifftshift(np.arange(nc))[:, None] -
    #                np.floor(nc / 2)).dot(
    #                   np.arange(upsampc)[None, :] - offsc))
    
    kernr = np.exp((-1j * 2 * np.pi / (nr * upsample_factor)) *
                   (np.arange(upsampr)[:, None] - offsr).dot(
                       np.fft.ifftshift(np.arange(nr))[None, :] -
                       np.floor(nr / 2)))
    
    kernc = np.exp((-1j * 2 * np.pi / (nc * upsample_factor)) *
                   (np.arange(upsampc)[:, None] - offsc).dot(
                       np.fft.ifftshift(np.arange(nc))[None, :] -
                       np.floor(nc / 2)))

    kernz = np.exp((-1j * 2 * np.pi / (nz * upsample_factor)) *
                   (np.arange(upsampz)[:, None] - offsz).dot(
                       np.fft.ifftshift(np.arange(nz))[None, :] -
                       np.floor(nz / 2)))
    
    out = np.reshape(kernr @ data.reshape(nr,-1), [upsampr, nc, nz])
    out = np.transpose(out, [1, 0, 2])
    out = np.reshape(kernc @ out.reshape(nc, -1), [upsampc, upsampr, nz])
    out = np.transpose(out, [1, 0, 2])
    out = np.transpose(out, [2, 1, 0])
    out = np.reshape(kernz @ out.reshape(nz, -1), [upsampc, upsampr, upsampz])
    out = np.transpose(out, [2, 1, 0])
    return out



def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be
        zero if images are non-negative).
    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)



def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    Compute RMS error metric between ``src_image`` and ``target_image``.
    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    src_amp : float
        The normalized average image intensity of the source image
    target_amp : float
        The normalized average image intensity of the target image
    """
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() / (src_amp * target_amp)
    return np.sqrt(np.abs(error))



def register_translation_3d(src_image, target_image, upsample_factor=1,
                         space="real"):
    """
    Efficient subpixel image translation registration by cross-correlation.
    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.
    Parameters
    ----------
    src_image : ndarray
        Reference image.
    target_image : ndarray
        Image to register.  Must be same dimensionality as ``src_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)
    space : string, one of "real" or "fourier"
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.
    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).
    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    
    ndim = src_image.ndim
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_image = np.array(src_image, dtype=np.complex128, copy=False)
        target_image = np.array(target_image, dtype=np.complex128, copy=False)
        src_freq = np.fft.fftn(src_image)
        target_freq = np.fft.fftn(target_image)
    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    upsample_factor = np.array(upsample_factor, dtype=np.float64) #ASI: added to avoid overflow when computing upsample_factor**2
    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = (np.fft.ifftn(image_product)) #ASI: removed fftshift
    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape) #ASI: np.argmax does not use abs of input, but it should here
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    #ASI: added the following 2 if statements
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
        target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5) #ASI: Increased default window size, single pixel precision guess might not be close enough to peak
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        normalization = (np.product(np.asarray(shape)) * np.asarray(upsample_factor) ** 2) #ASI: changed normalization (not necessary)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor #ASI: changed sample region offset
        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj() #ASI: conjuagte input to upsampled dft
        cross_correlation /= normalization

        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(np.argmax(np.abs(cross_correlation)),
                                           cross_correlation.shape),
                          dtype=np.float64) #ASI: np.argmax does not use abs of input, but it should here
        maxima -= dftshift
        shifts = shifts + maxima / upsample_factor #ASI: changed sign of pixel shift adjustment
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1,
                                    upsample_factor)[0, 0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    shifts = np.fix(shifts)
    for dim in range(ndim):
        if midpoints[dim] == 1:
            shifts[dim] = 0
    return shifts, _compute_error(CCmax, src_amp, target_amp), _compute_phasediff(CCmax)

