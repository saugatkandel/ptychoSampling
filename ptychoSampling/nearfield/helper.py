#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import scipy
import skimage, skimage.transform, skimage.data
from skimage.feature import register_translation
from typing import NamedTuple, Optional, List, Tuple
import tensorflow as tf
from pandas import DataFrame
import dataclasses as dt
import matplotlib.pyplot as plt



def getRandomComplexArray(shape: Tuple[int,...],
                          mod_range: float = 1.0,
                          phase_range: float = np.pi) -> np.ndarray:
    rands = np.random.random((2, *shape))
    return rands[0] * mod_range * np.exp(1j * rands[1] * phase_range)



def getSampleObj(npix: int = 256, 
                 mod_range: float = 1, 
                 phase_range: float = np.pi)-> np.ndarray:
    r"""Creates a sample complex-valued object using stock data from the skimage library.
    
    Uses the stock camera image for the phase and the stock immunohistochemistry image (channel 0) for the modulus [1]_.
    
    Parameters
    ----------
    npix : int
        Number of pixels in each axis of the object
    mod_range : float 
        Maximum value of the modulus for the object pixels.
    phase_range : float
        Maximum value of the phase for the object pixels.

    Returns
    -------
    out : ndarray(complex)
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



def getGaussian2D(npix: int, 
                  stdev: Optional[float] = None,
                  fwhm: Optional[float] = None) -> np.ndarray:
    r"""Generates a circularly symmetric 2d gaussian pattern. 
    
    Ignoring the scaling constants, a circularly symmetric 2d gaussian can be calculated as:
    
    .. math:: g = \exp\left(-\frac{(x - c_x)^2 + (y - c_y)^2}{2 * stdev^2}\right)
    
    where :math:`(x,y)` are the coordinate indices, :math:`(c_x, c_y)` are the coordinates for the center of the gaussian, and stdev is the standard deviation. In this function, we assume that :math:`c_x = c_y = npix // 2` (i.e. the center of the gaussian is the center of the 2d array).
    
    Parameters
    ----------
    npix : int
        Number of pixels in each axis of the probe
    stdev : float, optional
        Standard deviation of the gaussian. The function requires either the standard deviation or the fwhm. 
    fwhm : float, optional
        Full width at half maximum (FWHM) of the peak. The function requires either the standard deviation or the fwhm. If we supply the fwhm, the standard deviation is calculated as :math:`stdev = fwhm / 2.35682`.
    
    Returns
    -------
    out : ndarray(float)
        A 2d array of shape :math:`n_{pix}\times n_{pix}` and dtype `float32`.
    """
    
    
    if ((stdev is None) and (fwhm is None)) or ((stdev is not None) and (fwhm is not None)):
        raise KeyError("Should input only one of either stdev or fwhm.")
    if fwhm:
        stdev = fwhm / 2.35682
    
    center = npix // 2
    xvals = np.arange(npix)
    XX, YY = np.meshgrid(xvals, xvals)
    r_squared = (XX - center)**2 + (YY - center)**2
    
    # Ignoring the normalization constants
    gaussian = np.exp(-r_squared/ (2 * stdev**2)) 
    return gaussian.astype('float32')



def getNFPropKernel(beam_shape: tuple, 
                    pixel_pitch: float, 
                    wavelength: float, 
                    prop_dist: float,
                    fftshift: bool = True) -> np.ndarray:
    r"""Generates a kernel for near-field wavefield propagation using the Transfer function method. 
    
    This implementation is adapted slightly the reference algorithm in [2]_. The expression used is:
    
    .. math:: H(f_x, f_y) = \exp\left(-j \pi \lambda z (f_x^2 + f_y^2)\right)
    
    where :math:`(f_x, f_y)` are the coordinates in reciprocal space (which are calculated using the pixel pitch), :math:`z` is the propagation distance, and :math:`\lambda` is the wavelength.
    
    Note that this is equivalent to the angular spectrum propagator used in Equation 14 in [3]_ , which is itself sourced from Equation 4 in [4]_. In these equations, we see the coordinates :math:`(q_x, q_y)` and :math:`(u, v)`---these are the *angular* coordinates, which correspond to the *frequency* coordinates used herein, i.e. :math:`(q_x = u = 2 * \pi * f_x)`.
    
    Parameters
    ----------
    beam_shape : tuple(int, int)
        A tuple for the shape of the array containing the propagation kernel.
    pixel_pitch : float
        Pixel pitch at plane containing the initial wavefront (in SI units).
    wavelength : float
        Wavelength of the propagated wave (in SI units).
    prop_dist : float
        Propagation distance (in SI units).
    fftshift : bool
        Whether to return the kernel after performing an fftshift. Since we most often use the output kernel (H) inside the structure :math:`IFFT(FFT(\psi)\cdot H)` where :math:`psi` is the wavefront that we need to propagate, it is convenient to deal with the shifted kernel directly. 
    
    Returns
    --------
    out : ndarray(complex)
        A 2d array of shape `beam_shape` and dtype complex64.
    
    References
    ----------
    .. [2] Chapter 5.1 of the book "Computational Fourier Optics: A MATLAB Tutorial" by David Voelz (2011).
    .. [3] Saugat Kandel, S. Maddali, Marc Allain, Stephan O. Hruszkewycz, Chris Jacobsen, and Youssef S. G. Nashed, "Using automatic differentiation as a general framework for ptychographic reconstruction," Opt. Express 27, 18653-18672 (2019)
    .. [4] Richard M. Clare, Marco Stockmar, Martin Dierolf, Irene Zanette, and Franz Pfeiffer, "Characterization of near-field ptychography," Opt. Express 23, 19728-19742 (2015)
    """
    
    M, N = beam_shape

    # Getting the reciprocal space cordinates
    x = np.arange(-N // 2, N // 2) / (N * pixel_pitch)
    y = np.arange(-M // 2, M // 2)[:, np.newaxis] / (N * pixel_pitch)

    H = np.exp(-1j * np.pi * (x ** 2 + y ** 2) * (wavelength * prop_dist))

    if fftshift: 
        H = np.fft.fftshift(H)
    return H.astype('complex64')



def getSpeckle(npix: int, 
               window_size: int) -> np.ndarray:
    r"""Generates a speckle pattern. 
    
    To generate a speckle pattern, this function uses a ``window_size x window_size`` array of complex numbers with unit amplitude and uniformly random phase. This array is padded with zeros to get an ``npix x npix`` array, an FFT of which gives us a speckle pattern. The speckle pattern thus generated is discontinuous; there is a phase step of pi between adjacent pixels in both the ``x`` and ``y`` directions. We remove these discontinuities to get the final, continuous, speckle pattern.
    
    Parameters
    ----------
    npix : int
        Number of pixels along each side of the 2d array containing the speckle pattern.
    window_size : int 
        The size of the rectangular window used to generate the speckle pattern. Larger window sizes give smaller speckle sizes and vice versa. (*Note*: I tried a circular window as well, but the results did not change 
    noticeably.)
    
    Returns
    --------
    out : ndarray(complex)
        A 2d array of size ``npix x npix`` and dtype complex64.
    """
    
    if window_size > npix: 
        raise ValueError("Window size should be smaller than the size of output array.")
    
    # generating the random array
    ran = np.exp(1j * np.random.rand(npix,npix) * 2 * np.pi)
    
    window = np.zeros((npix, npix))
    indx1 = npix // 2 - window_size // 2
    indx2 = npix // 2 + window_size // 2
    window[indx1: indx2, indx1: indx2] = 1
    
    # Circular window - doesn't change the results.
    #xx, yy = np.meshgrid(np.arange(npix), np.arange(npix))
    #mask = ((xx - npix // 2)**2 + (yy - npix // 2)**2 < (window_size // 2)**2)
    #window[mask] = 1
    
    t = window * ran
    
    ft = np.fft.fftshift(np.fft.fft2(t, norm='ortho'))
    absvals = np.abs(ft)
    angvals = np.angle(ft)
    
    # Removing the discontinuities in the phases 
    angvals[::2] = (angvals[::2] + np.pi) % (2 * np.pi)
    angvals[:,::2] = (angvals[:, ::2] + np.pi) % (2 * np.pi)
    return (absvals * np.exp(1j * angvals)).astype('complex64')



@dt.dataclass(frozen=True)
class ObjParams:
    r"""Convenience class to store the object parameters.
    
    This is frozen after creation.
    
    Notes
    -----
    Adding an assumed-known border (filled with ones) to the object helps avoid the affine phase ambiguity.
    
    Parameters
    ----------
    npix : int
        Number of pixels in each side of the (square) object to be generated.
    mod_range : float, optional
        Maximum value of the modulus for the object pixels. Defaults to `None`.
    phase_range : float, optional
        Maximum value of the phase for the object pixels. Defaults to `None`.
    border_npix : int, optional
        Number of pixels to add along the border in the left, right, top, and bottom margins of the object. If ``border_npix = 10``, then the simulation adds 10 pixels to the left of the object, and 10 pixels to the right of the object, i.e. a total of 20 pixels along the x-direction (and similarly with y). Defaults to ``0``. 
    border_const : float
        Constant value to fill the border with. Defaults to ``0``. 
        
    Attributes
    ----------
    npix, mod_range, phase_range, border_npix, border_const: see Parameters
    
    obj_w_border_npix : int
        Number of pixels in the extended object array. ``obj_w_border_npix = 2 * border_npix + npix``.
    """
    npix: int
    mod_range: Optional[float] = None
    phase_range: Optional[float] = None
    border_npix: int = 0
    border_const: float = 0
    obj_w_border_npix: int = dt.field(init=False)
    
    def __post_init__(self):
        """Total number of pixels (including object and border) in each side of the object."""
        object.__setattr__(self, 'obj_w_border_npix', self.npix + 2 * self.border_npix)   

@dt.dataclass(frozen=True)
class ProbeParams:
    r"""Convenience class to store the probe parameters.

    Parameters
    ----------
    wavelength : float
        Wavelength of the probe beam.
    npix : int
        Number of pixels in each side of the square probe to be generated.
    n_photons : float
        Total number of photons.
        
    Attributes
    ----------
    wavelength, npix, n_photons : see Parameters
    photons_flux : float
        Average number of photons per pixel in the probe beam. ``photons_flux = n_photons / npix**2``
    """
    wavelength: float
    npix: int
    n_photons: float
    photons_flux: float = dt.field(init=False)
    
    def __post_init__(self):
        object.__setattr__(self, 'photons_flux', self.n_photons / self.npix**2)

@dt.dataclass(frozen=True)
class DetectorParams:
    r"""Convenience class to store the detector parameters.
    
    Parameters
    -----------
    obj_dist : float
        Object-detector distance (in m).
    pixel_pitch : float
        Width of each individual pixel (in m). 
        
    Attributes
    ----------
    obj_dist, pixel_pitch : see Parameters
    """
    obj_dist: float
    pixel_pitch: float

@dt.dataclass(frozen=True)
class ScanParams:
    r"""Convenience class to store the ptychographic scan parameters.

    Parameters
    ----------
    scan_step_npix : int
        Number of pixels per step in the raster grid.
    poisson_noise : bool
        Whether to simulate Poisson noise in the diffraction data.
    
    Attributes
    ----------
    scan_step_npix, poisson_noise : see Parameters
    """
    scan_step_npix: int
    poisson_noise: bool



@dt.dataclass(frozen=True)
class NFSimObjParams(ObjParams): 
    r"""Customizing `ObjParams` with some default parameters for nearfield simulations.
    
    Only function is to use the default parameters. Does not extend `ObjParams`. 
    
    See `ObjParams` for documentation on the individual parameters used.
    
    See also
    --------
    ObjParams
    """
    npix : int = 192
    mod_range: float = 1.0
    phase_range: float = np.pi
    border_npix: int = 32
    border_const: float = 1.0

@dt.dataclass(frozen=True)
class NFSimDetectorParams(DetectorParams):
    r"""Customizing `DetectorParams` with some default parameters for nearfield simulations.
    
    Only function is to use the default parameters. Does not extend `DetectorParams`. 
    
    See `DetectorParams` for documentation on the individual parameters and attributes used.
    
    See also
    --------
    DetectorParams
    """
    obj_dist: float = 0.0468
    pixel_pitch: float = 3e-7

@dt.dataclass(frozen=True)
class NFSimProbeParams(ProbeParams):
    r"""Customizes `ProbeParams` with default parameter values and also adds parameters specific to the nearfield simulation.
    
    For detail on the other attributes and parameters, refer to the documentation for the `ProbeParams` class.
        
    Parameters
    ----------
    gaussian_intensity_stdev_pix : float
        Standard deviation of the gaussian probe to be used. See the function `getGaussian2D` for more detailed information on this parameter.
    speckle_window_pix : int
        The size of the rectangular window used to generate the speckle pattern. See the function `getSpeckle` for more detail on this parameter.
        
    Attributes
    ----------
    gaussian_intensity_stdev_npix, speckle_window_npix : see Parameters.
    
    See also
    --------
    ProbeParams
    """
    wavelength: float = 0.142e-9
    npix: int = 512
    photons_flux: float = 1e4
    gaussian_intensity_stdev_npix: float = 150.0
    speckle_window_npix: int = 40

@dt.dataclass(frozen=True)
class NFSimScanParams(ScanParams):
    r""""Customizes `ScanParams` with default parameter values and also adds parameters specific to the nearfield simulation.
    
    For detail on the other parameters, refer to the documentation for the `ScanParams` class.
        
    Parameters
    ----------
    scan_area_buffer_npix : int
        This is a bit of hack to ensure that the scan area is around the center of the full-field probe. For e.g., if ``scan_area_buffer_npix = 20``, we start the ptychographic scan from the coordinate :math:`(20, 20)` of the probe instead of from :math:`(0,0)`. The regions between :math:`x=[0,20]` and :math:`y=[0,20]` are never sampled. Basically, this is like imposing a margin of ``20`` pixels in the left and bottom of the probe.
    
    Attributes
    ----------
    scan_area_buffer_npix : see Parameters
    
    See also
    --------
    ScanParams
    """
    scan_step_npix: int = 44
    poisson_noise: bool = True
    scan_area_buffer_npix: int = 20



class NFPtychoSimulation(object):
    r"""Simulate a near-field ptychography simulation using a full-field probe and object translations.
        
    I don't know if this is the typical scenario for a near-field ptychography experiment. For this work, I am trying to use a similar setup to that used by Clare et al in [6]_.

    See `NFSimObjParams`, `NFSimProbeParams`, `NFSimDetectorParams`, and `NFSimScanParams` classes for details on the options available to customize the simulation.
    
    Parameters
    -----------
    obj_args : dict
        Dictionary that contains pairs of arguments and values to use to create a custom `NFSimObjParams` class, and thus change the parameters of the simulated object. 
    probe_args : dict
        Dictionary that contains pairs of arguments and values to use to create a custom `NFSimProbeParams` class, and thus change the parameters of the simulated probe. 
    detector_args : dict
        Dictionary that contains pairs of arguments and values to use to create a custom `NFSimDetectorParams` class, and thus change the parameters of the detector used. 
    scan_args : dict
        Dictionary that contains pairs of arguments and values to use to create a custom `NFSimScanParams` class, and thus change the parameters of the ptychographic scan. 
    
    Attributes
    ----------
    obj : ndarray(complex)
        Holds the generated complex valued object.
    obj_w_border : ndarray(complex)
        Holds the generated complex valued object along with the border specified through `obj_args`.
    probe: ndarray(complex)
        Holds the generated complex valued probe.
    positions : ndarray(int, ndim=2)
        Holds the scan positions (object translations) used for the ptychographic scan.
    diffraction_mods : ndarray(float)
        Holds the modulus of the exit wave at the detector plane for the scan positions. This is just the square root of the intensity pattern at the object plane.
    ndiffs : int
        Number of diffraction patterns.
    _obj_params : NFSimObjParams
        Parameters for the simulated object.
    _probe_params : NFSimProbeParams
        Parameters for simulated probe.
    _det_params : NFSimDetectorParams
        Parameters for simulated detector.
    _scan_params : NFSimScanParams
        Parameters for the scan grid and noise setup.
    _prop_kernel: ndarray(complex)
        Holds the (fft-shifted) Fresnel propagation kernel for the simulation parameters. 
    
    Examples
    --------
    Input parameter examples: 
    
    * If we want to use an object 64 pixels wide, and with of border of 5 pixels on each side, we use: 
        ``nfsim = NFPtychoSimulation(obj_args={obj_npix:64, border_npix:5})``
    * If we want to use an object 1024 pixels wide, and with a wavelength of 1 nm, we use:
        ``nfsim = NFPtychoSimulation(probe_args={probe_npix:1024, wavelength:1e-9})``
    * If we want to use a detector with pixel pitch 100nm, we use:
        ``nfsim = NFPtychoSimulation(detector_args={pixel_pitch:100e-9})``
    * If we want to use a step size of 60 pixels, and generate diffraction patterns without Poisson noise included, we use:
        ``nfsim = NFPtychoSimulation(scan_args={scan_step_npix:60, poisson_noise=False})``

    References
    -----------
    .. [6] Richard M. Clare, Marco Stockmar, Martin Dierolf, Irene Zanette, and Franz Pfeiffer, "Characterization of near-field ptychography," Opt. Express 23, 19728-19742 (2015).
    
    See also
    --------
    NFSimObjParams
    NFSimProbeParams
    NFSimDetectorParams
    NFSimScanParams
    """
    def __init__(self,
                 obj: Optional[np.ndarray] = None,
                 probe: Optional[np.ndarray] = None,
                 obj_args: dict = {},
                 probe_args: dict = {},
                 detector_args: dict = {},
                 scan_args: dict = {}) -> None:
  
        self._obj_params = NFSimObjParams(**obj_args)
        self._probe_params = NFSimProbeParams(**probe_args)
        self._det_params = NFSimDetectorParams(**detector_args)
        self._scan_params = NFSimScanParams(**scan_args)
        

        self._checkValidity()
        
        if obj is None:
            obj = getSampleObj(npix=self._obj_params.npix,
                                mod_range=self._obj_params.mod_range,
                                phase_range=self._obj_params.phase_range)
        self.obj = obj
        pad = self._obj_params.border_npix
        self.obj_w_border = np.pad(self.obj, [[pad, pad], [pad,pad]],
                                   mode='constant',
                                   constant_values=self._obj_params.border_const)
        
        if probe is None:
            gaussian_intensity = getGaussian2D(self._probe_params.npix, 
                                               self._probe_params.gaussian_intensity_stdev_npix)
            gaussian_ampl = gaussian_intensity**0.5
            speckle = getSpeckle(self._probe_params.npix, 
                                 self._probe_params.speckle_window_npix)
            probe_data = gaussian_ampl * speckle
            probe = probe_data * np.sqrt(self._probe_params.n_photons / np.sum(np.abs(probe_data)**2))
        self.probe = probe
        
        # Generating the fft-shifted propagation kernel 
        self._prop_kernel = getNFPropKernel(beam_shape=self.probe.shape,
                                           pixel_pitch=self._det_params.pixel_pitch,
                                           wavelength=self._probe_params.wavelength,
                                           prop_dist=self._det_params.obj_dist)
        
        self.positions = self._getPtychographyPositions()
        self.diffraction_mods = self._getDiffractionMods()
        self.ndiffs = self.diffraction_mods.shape[0]    
        
    def _checkValidity(self) -> None:
        r"""Checking if the parameters supplied are valid. 
        
        For now, we only check to ensure that the step size supplied is larger than the 
        width of the Fresnel zone for the simulation parameters. This ensures that the 
        generated diffraction patterns have enough diversity.
        
        Need to add more checks here.
        """
        
        fresnel_zone_dist = np.sqrt(self._probe_params.wavelength * self._det_params.obj_dist)
        fresnel_zone_npix = fresnel_zone_dist / self._det_params.pixel_pitch
        
        error_str = (f"Step size ({self._scan_params.scan_step_npix} is too small. "
                     + f"Ensure that the step size is at least larger than the Fresnel zone width "
                     + f"({fresnel_zone_npix}) to ensure diversity in the diffraction patterns.")
        assert self._scan_params.scan_step_npix > fresnel_zone_npix, error_str  
    
    def _getPtychographyPositions(self) -> np.ndarray:
        r"""Generate the scan positions for the ptychographic scan using a square raster grid.
        
        Returns
        -------
        out : ndarray(int, ndim=2)
            Array of scan positions
        """
        
        p1 = self._scan_params.scan_area_buffer_npix
        p2 = self._probe_params.npix - p1 - self._obj_params.obj_w_border_npix
        positions_x = np.arange(p1, p2, self._scan_params.scan_step_npix)
        positions = []
        
        for r in positions_x:
            for c in positions_x:
                positions.append([r,c])
        return np.array(positions)
    
    def _getDiffractionMods(self) -> np.ndarray:
        r"""Generate the near-field diffraction patterns for the ptychography scan using the transfer function method.
        
        Returns
        -------
        out: ndarray(float, ndim=3)
            Array containing the 2d diffraction patterns stacked in order of the ptychographic scan positions.
        """
        diffraction_intensities = []
        
        npix_pad = self._probe_params.npix - self._obj_params.obj_w_border_npix
        obj_padded_to_probe = np.pad(self.obj_w_border, 
                                     [[0, npix_pad], [0, npix_pad]],
                                     mode='constant',
                                     constant_values=1.0)
        for indx, (r,c) in enumerate(self.positions):
            exit_wave = self.probe * np.roll(obj_padded_to_probe, [r,c], axis=(0,1)) 
            nearfield_wave = np.fft.ifftshift(np.fft.ifft2(self._prop_kernel * np.fft.fft2(exit_wave)))
            diffraction_intensities.append(np.abs(nearfield_wave)**2)
            
        if self._scan_params.poisson_noise: 
            diffraction_intensities = np.random.poisson(diffraction_intensities)
        return np.sqrt(diffraction_intensities)
        
    def _getScatterIndices(self) -> None:
        r"""Not in use right now"""
        scatter_indices_all = []
        for py, px in self.positions:
            R, C = np.ogrid[py:self._obj_params.obj_w_border_npix + py, 
                            px:self._obj_params.obj_w_border_npix + px]
            scatter_single = ((R % self._probe_params.npix) * self._probe_params.npix + 
                              (C % self._probe_params.npix))
            scatter_indices_all.append(scatter_single)
        scatter_indices =  np.array(scatter_indices_all)
        return scatter_indices



@dt.dataclass(frozen=True)
class ReconsInits:
    r"""Holds the input position and diffraction data.
    
    Can be extended to hold additional input data (such as angles) by subclassing.
    
    Parameters
    -----------
    positions : array_like(int, ndim=2)
        Ptychographic scan positions. The coordinates indicate the position of the :math:`(0,0)` corner of the probe or object, whichever is translated.
    mods : array_like(float)
        Measured diffraction data corresponding to each of the ptychographic scan positions. Contains the modulus of the exit wave at the detector plane (i.e., the square root of the recorded intensities).
    obj_params : ObjParams
        Set of parameters that determine the extent and border of the reconstructed object. See the `ObjParams` documentation for more detail.
    probe_params : ProbeParams
        Set of parameters that determine the extent of the reconstructed probe. See the `ProbeParams` documentation for more detail.
    det_params : DetectorParams
        Set of parameters that determine the properties of the detector used, viz. the object detector distance and the pixel pitch of the detector.
    obj : array_like(complex), optional
        Initial guess  of the object. Uses a random complex array as default if the guess is not user-supplied.
    probe : array_like(complex), optional
        Holds either the initial guess of the probe (if we are reconstructing the probe). If not supplied by the user, the default guess is calculated by backpropagating the mean diffraction pattern.
        
    Attributes
    ----------
    positions, mods, obj_params, probe_params, det_params, obj, probe : see Parameters
    
    mods_shifted : array_like(float)
        Measured diffraction data rearranged (for convenience) using an fftshift along the detector axes. This is generated automatically.
    prop_kernel : array_like(complex)
        Holds the calculated Fresnel propagation kernel for the parameters supplied. This is generated automatically.
        See the `getNFPropKernel` documentation for more detail.
    """
    positions: np.ndarray
    mods: np.ndarray
    obj_params: ObjParams
    probe_params: ProbeParams
    det_params: DetectorParams
    obj: np.ndarray = None
    probe: np.ndarray = None
    mods_shifted: np.ndarray = dt.field(init=False)
    prop_kernel: np.ndarray = dt.field(init=False)
    
    def __post_init__(self):
        r"""Uses the user-supplied parameters to calculate the fftshifted diffraction patterns, the Fesnel propagation kernel, and (if necessary) the probe and object guesses.
        
        The propagation kernel is calculated using the transfer function method, then rearranged with an fftshift for ease of usage.
        
        The default object guess is a random complex array.
        
        The default probe guess is calculated by backpropagating the average diffraction pattern.
        """
        
        mods_shifted = np.fft.fftshift(self.mods, axes=(-1,-2))

        opix = self.obj_params.npix
        if self.obj is None:
            obj = getRandomComplexArray((opix, opix))
            object.__setattr__(self, 'obj', obj)
        
        ppix = self.probe_params.npix
        prop_kernel = getNFPropKernel(beam_shape=(ppix, ppix),
                                             pixel_pitch=self.det_params.pixel_pitch,
                                             wavelength=self.probe_params.wavelength,
                                             prop_dist=self.det_params.obj_dist)
        if self.probe is None:
            mods_avg = np.mean(mods_shifted, axis=0)
            probe = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(mods_avg) 
                                                  / prop_kernel))
            object.__setattr__(self, 'probe', probe)
        object.__setattr__(self, 'prop_kernel', prop_kernel)
        object.__setattr__(self, 'mods_shifted', mods_shifted)
    
@dt.dataclass(frozen=True)
class ForwardModel:
    r"""Collection of inputs/outputs that represent the forward model for a given minibatch.
    
    A ``minibatch`` here represents a subset of the collected dataset (i.e., the diffraction patterns). For example, if we have a total of 8 individual diffraction patterns (or 16 scan positions), then a minibatch of size :math:`b` with :math:`b\leq 16` can contain one or more of these diffraction patterns. 
    
    To keep track of which diffraction patterns each minibatch contains, we represent each diffraction pattern by its `index` in the overall dataset. For example, suppose our diffraction patterns are collected as:
    ``diff_patterns = [A, B, C, D, E, F, G, H]``
    where each letter represents a diffraction pattern, and we have a minibatch
    ``minibatch_patterns = [B, G, H].``
    Then, we can track the diffraction patterns in `minibatch_patterns` by their relative indices in `diff_patterns`,
    ``indices = [1, 6, 7].``
    As long as the diffraction patterns and the scan positions (and any other relevant information) are both arranged in the same sequence, then using these `indices` makes the minibacth calculations more convenient.
    
    Parameters
    ----------
    ndiffs : int
        Number of diffraction patterns (or scan positions) that make up the minibatch.
    indices_t : array_like(int)
        The indices of the diffraction patterns relative to the overall diffraction data set.
    mods_t : tensor(float)
        Tensor containing the diffraction patterns that make up the minibatch.
    obj_views_t : tensor(complex)
        Tensor containing the object slices that correspond to each diffraction patterns. The transfer function represented by the object slices changes automatically every time we update the object variable.
    predictions_t : tensor(float), optional
        Tensor containing the predicted diffraction patterns for each object slices. The predicted data changes with every change in the object and probe variables. This is an optional parameter, thus accounting for the case when we do want a collection of object slices, but do not actually want the forward propagation for such a minibatch.
        
    Attributes
    ----------
    ndiffs, indices_t, mods_t, obj_views_t, predictions_t : see Parameters 
    
    .. todo::

        Instead of a data class, make this a full-fledged class that handles the propagation.
    """
    ndiffs: int
    indices_t: tf.Tensor
    mods_t: tf.Tensor
    obj_views_t: tf.Tensor
    predictions_t: Optional[tf.Tensor] = None

@dt.dataclass
class ReconsLossAndOptimizers:
    r"""Collection of objects required for the optimization.
    
    Contains separate optimizers (and minimize ops) for the object and probe variables. Since the probe values are usually much higher than the object values, creating separate optimizers with different learning rates helps accelerate the reconstruction procedure. Using the same optimizer (i.e., the same learning rate) for differently scaled variables leads to very slow convergence.  
    
    Parameters
    ----------
    batch_loss_t : tensor(float)
        Tensor that calculates/stores the training loss value per minibatch.
    validation_loss_t : tensor(float)
        If the dataset is split up into training/validation, then this parameter calculates/stores the validation loss. Should be set to zero otherwise.
    obj_learning_rate : float
        Learning rate (or descent step size) for the object variable.
    obj_opt : tensorflow Optimizer
        Tensorflow optimizer defined with the learning rate `obj_learning_rate` for use for the object variable only. 
    obj_min_op : tensorflow op
        Tensorflow op that invokes the minimization step for the object variable. Is derived from the `obj_opt` object. Should be invoked with a ``session.run(obj_min_op)`` command (or variations thereof)---each such call updates the object variable *once*.
    probe_learning_rate : float
        Learning rate (or descent step size) for the probe variable. Default ``0`` value is used when we are not reconstructing the probe function. 
    probe_opt : tensorflow optimizer, optional
        Tensorflow optimizer defined with the learning rate `obj_learning_rate` for use for the object variable only. Needs to be defined only when we want to reconstruct the probe function.
    probe_min : tensorflow op
        Tensorflow op that invokes the minimization step for the probe variable. Needs to be defined only when we want to reconstruct the probe function.
        
    Attributes
    ----------
    batch_loss, validation_loss, obj_learning_rate, obj_opt, obj_min_op, probe_learning_rate, probe_opt, probe_min : see Parameters
    
    """
    batch_loss_t: tf.Tensor
    validation_loss_t: tf.Tensor
    obj_learning_rate: float
    obj_opt: tf.train.Optimizer
    obj_min_op: tf.Tensor
    probe_learning_rate: float = 0
    probe_opt: Optional[tf.train.Optimizer] = None
    probe_min_op: Optional[tf.train.Optimizer] = None
    
@dt.dataclass
class ReconsOutputs:
    r"""Collection of outputs from the optimization procedure.
    
    Parameters
    ----------
    obj : ndarray(complex)
        Numpy array containing the current (after optimization) estimate of the object function.
    probe : ndarray(complex)
        Numpy array containing the current (after optimization) estimate of the probe function.
    log : Dataframe
        Pandas Dataframe containing metrics (training loss, validation loss, epoch, etc) used to characterize the current step of the optimization procedure. This should be updated at every iteration.
    
    Attributes
    ----------
    obj, probe, log : see Parameters
    
    """
    obj: np.ndarray
    probe: np.ndarray
    log: DataFrame



class tfNFPtychoRecons(object):
    r"""Reconstruct the object and probe from a near-field ptychography simulation. 

    Assumes that the probe is full-field, the object is (much) smaller than the probe, and the object is translated within the probe field to generate the diffraction patterns. 

    Assumes square object and probe.

    Uses the gaussian noise model for the loss function. This is easy to change if neeeded.
    
    This implementation assumes that the diffraction data set is small enough to fit in the GPU memory. Need to modify the minibatch scheme otherwise.

    Parameters
    ----------
    positions : array_like(int, ndim=2)
        Ptychographic scan positions, indexed by pixel number.
    diffraction_mods : array_like(float, ndim=3)
        Array containing the 2d diffraction patterns (moduli). Square root of the measured diffraction intensities.  
    det_args : dict
        Dictionary containing the object-detector distance and the detector pixel pitch. See `DetectorParams`.
    probe_args : dict
        Dictionary containing the necessary parameters to create a `ProbeParams` instance.
    obj_args : dict
        Dictionary containing the necessary parameters to create an `ObjParams` instance. The ``mod_range`` and ``phase_range`` parameters in `ObjParams` are not currently used during the reconstruction and can be safely ignored.
    obj_guess : array_like(complex, ndim=2), optional
        Initial guess for the object. If supplied, this takes precendence over ``npix`` in `obj_args`.
    probe_guess : array_like(complex, ndim=2), optional
        Initial guess for the probe. If supplied, this takes precdedence over ``npix`` in `probe_args`. If we do not want to reconstruct the probe function (i.e., if the probe function is priorly known), we need to supply the known probe function for `probe_guess` and set `probe_recons` to False.
    probe_recons : bool, optional
        Whether to reconstruct the probe function. Defaults to True.
    batch_size : int, optional
        Minibatch size to use for the reconstruction procedure. Defaults to `0`, which sets the entire training dataset as a single minibatch. See the documentation for `_createBatches` for more detail.
    n_validation_diffs : int
        Number of (randomly selected) diffraction patterns to put aside as *validation* data to be used as early stopping criterion. The training dataset then contains `n_diffs_total - n_validation_diffs` diffraction patterns.
    obj_true : array_like(complex, ndim=2), optional
        Ground truth for the object variable (if known). Is used to calculate the object reconstruction error at specified intervals during the reconstruction procedure.
    probe_true : array_like(complex, ndim=2), optional
        Ground truth for the probe variable (if known). Is used to calculate the probe reconstruction error at specified intervals during the reconstruction procedure.
    
    Attributes
    ----------
    probe_recons, obj_true, probe_true, batch_size : see Parameters
    graph : TensorFlow Graph
        The Graph object for all the tensorflow calculations within an instance of `tfNFPtychoRecons`.
    session : TensorFlow Session
        The Session object for all the tensorflow calculations within this instance of `tfNFPtychoRecons`.
    outs : ReconsOutputs
        Dataset that holds the current object and probe guesses in numpy ndarray form (instead of as tensors) as well as the pandas Datafrom that contains the reconstruction metrics. See also the documentation for ReconsOutputs.
    inits : ReconsInits
        Dataclass that holds the supplied ptychographic data (positions and diffraction patterns) as well as the supplied object and probe parameters and initial guesses. If no initial guess is supplied, then the initial guess is generated automatically. See also the documentation for ReconsInits.
    _obj_v : tf.Variable
        Tensor that separately contains the real and imaginary parts of the object variable. See the second case in the example above.
    _obj_cmplx_t : tf.Tensor
        Tensor that holds the object variable as complex numbers. 
    _obj_w_border_t : tf.Tensor
        Tensor that holds the object with the border added. Could be identical to obj_cmplx if we do not add a border to the object.
    _probe_v : tf.Variable
        Tensor that separately contains the real and imaginary parts of the probe variable.
    _probe_cmplx_t : tf.Tensor
        Tensor that holds the probe variable as complex numbers.
    _prop_kernel_t : tf.Tensor
        Tensor that holds the near-field propagation kernel after the fftshift. 
    _train_full : ForwardModel
        Dataclass that holds the tensors that represent the training data set. Contains the measured diffraction patterns, the corresponding object slices, and *indices* that identify the position of these object slices and diffraction patterns relative to the overall (training + validation) data set. Does not, however, contain the predicted data for the current object guess---this is contained in the `ForwardModel` representing the training minibatches. See also the documentation for `ForwardModel`.
    _validation : ForwardModel
        Similar to `train_full`, this holds the tensors that represent the validation data set. Since the validation data set can be assumed to be fairly small, we do not divide it into minibatches. As such, we calculate the predicted data for the entire validation data set and store it within this `validation` dataclass.
    _batch_model : ForwardModel
        This holds the forward model (with predicted data) for the current minibatch. See the documentation for `_createBatches` in for more information.
    _dataset_indices : TensorFlow Dataset
        A TensorFlow Dataset object that is used to simplify the shuffling and batching of diffraction patterns after each iteration.
    _dataset_batch : TensorFlow Dataset
        A TensorFlow Dataset object, derived from `dataset_indices` that handles the minibatching.
    _new_batch_op : tensor
        A tensor that invokes the creation of a new minibatch. Should be used in a `session.run()` call.
     
    
    Examples
    --------
    Example usage for optimization::
        
            nfrecons = tfNFPtychoRecons(...)
            nfrecons.initLossAndOptimizers(...)
            nfrecons.initSession()
            nfrecons.run(...)
    
    See also
    --------
    DetectorParams
    ObjParams
    ProbeParams
    ReconsInits
    ReconsOutputs
    ForwardModel
    ReconsLossAndOptimizers
    
    Notes
    -----
    The default optimizers supplied with TensorFlow only accept real-valued variables. While it is not too difficult to write custom optimizers for complex-valued variables, we can also just use separate the real and imaginary parts of each complex-valued variable. This can be accomplished by either:
    
    * creating separate Variable objects for the real and imaginary parts of a complex-valued variable :math:`z`:
    
        .. code-block:: python
        
            z_reals = tf.Variable(np.real(z))
            z_imag = tf.Variable(np.imag(z))
            z = tf.complex(z_reals, z_imag)
        
        Then, for the optimization, we can ask the optimizer to use both the variables for the gradient descent:
        
        .. code-block:: python
            
            optimizer = tf.train.AdamOptimizer(learning_rate)
            minimizer = optimizer.minimize(loss_function, var_list=[z_reals, z_imag])
    
    * creating one single Variable object of size :math:`2N`. One of the many ways to do this is:
    
        .. code-block:: python
            
            z_reals = tf.Variable(np.array([np.real(z), np.imag(z)]))
            z = tf.complex(z_var[0], z_var[1])
            optimizer = tf.train.AdamOptimizer(learning_rate)
            minimizer = optimizer.minimize(loss_function, var_list=[z_reals])
            
    Both these structures are exactly equivalent. With just this much work, we can now use the optimizers.
    
    .. todo::
        
            Make it more convenient to attach/detach variables and optimizers. The current arrangement is a stopgap.
            
    """
    
    def __init__(self,
                 positions: np.ndarray,
                 diffraction_mods: np.ndarray,
                 det_args: dict,
                 obj_args: dict,
                 probe_args: dict,
                 obj_guess: Optional[np.ndarray] = None,
                 probe_guess: Optional[np.ndarray] = None,
                 probe_recons: bool = True,
                 batch_size: int = 0,
                 n_validation_diffs: int = 0,
                 obj_true: Optional[np.ndarray] = None,
                 probe_true: Optional[np.ndarray] = None) -> None:
                 
        det_params = self._setParams(DetectorParams, det_args)
        obj_args = self._checkGuessNpix(obj_args, obj_guess)
        probe_args = self._checkGuessNpix(probe_args, probe_guess)
        obj_params = self._setParams(ObjParams, obj_args)
        probe_params = self._setParams(ProbeParams, probe_args)
        
        self.inits = ReconsInits(positions=positions,
                                 mods=diffraction_mods,
                                 obj_params=obj_params, 
                                 probe_params=probe_params,
                                 det_params=det_params,
                                 obj=obj_guess,
                                 probe=probe_guess)
        
        self.probe_recons = probe_recons
        
        # Tensorflow setup
        self._createTFModels(n_validation_diffs)
        self._createBatches(batch_size)
        
        self.obj_true = obj_true
        self.probe_true = probe_true
        
        log = DataFrame(columns=['loss','epoch','obj_error','probe_error','validation_loss','patience'],
                              dtype='float32')
        self.outs = ReconsOutputs(np.zeros_like(obj_guess),
                                  np.zeros_like(probe_guess),
                                  log)
    
    @staticmethod
    def _checkGuessNpix(args_dict: dict, 
                        guess: np.ndarray) -> dict:
        """Ensure that the number of pixels is supplied for the object and probe inits."""
        if guess is not None:
            print(f'Supplied guess overrides any npix value in {args_dict}.')
            args_dict['npix'] = guess.shape[0]
        elif ('npix' not in args_dict):
            raise KeyError(f'Need to supply either the guess or npix (in {args_dict}).')
        return args_dict
    
    @staticmethod
    def _setParams(params_dataclass: object,
                   args: dict) -> object:
        """Ignore any extra parameters supplied as an argument to initialize the pramams_dataclass"""
        fields = [f.name for f in dt.fields(params_dataclass) if f.init==True]
        args_filtered = {k: v for k,v in args.items() if k in fields}
        return params_dataclass(**args_filtered)
        
    def _createTFModels(self, n_validation_diffs: int) -> None:
        """Create the TensorFlow Graph, the object and probe variables, and the training and validation forward models.
        
        Creates: 
            * the Tensorflow graph object for the tensorflow calculations.
            * real-valued object and probe variables for the optimization procedure (see documentation for ReconsVarsAndConsts for more detail)
            * creates the complex-valued object and probe tensors, and adds borders if necessary.
            * creates the propagation kernel.
            * Divides the data into training and validation sets depending on the `n_validation_diffs` parameter. 
        
        Parameters
        ----------
        n_validation_diffs : int
            Number of randomly selected diffraction patterns to allocate to the validation data set.
        """
        oguess = self.inits.obj
        pguess = self.inits.probe
        ndiffs = self.inits.positions.shape[0]
        n_train_diffs = ndiffs - n_validation_diffs
        self.n_train_diffs = n_train_diffs
        self.n_validation_diffs = n_validation_diffs
        
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._obj_v = tf.Variable(np.array([np.real(oguess), np.imag(oguess)]), dtype='float32')
            self._obj_cmplx_t = tf.complex(self._obj_v[0], self._obj_v[1])
            pad = self.inits.obj_params.border_npix
            self._obj_w_border_t = tf.pad(self._obj_cmplx_t, [[pad, pad], [pad, pad]], 
                                  constant_values=self.inits.obj_params.border_const)
            self._prop_kernel_t = tf.constant(self.inits.prop_kernel, dtype='complex64')
            
            self._probe_v = tf.Variable(np.array([np.real(pguess), np.imag(pguess)]), dtype='float32')
            self._probe_cmplx_t = tf.complex(self._probe_v[0], self._probe_v[1])
            
            self._mods_t = tf.constant(self.inits.mods_shifted, dtype='float32')
            self._obj_views_t = self._getObjViewsStack()
            
            all_indices_shuffled_t = tf.constant(np.random.permutation(ndiffs), dtype='int64')
            validation_indices_t = all_indices_shuffled_t[:n_validation_diffs]
            train_indices_t = all_indices_shuffled_t[n_validation_diffs:]
            
            
            train_mods_t = tf.gather(self._mods_t, train_indices_t)
            train_obj_views_t = tf.gather(self._obj_views_t, train_indices_t)
            
            validation_mods_t = tf.gather(self._mods_t, validation_indices_t)
            validation_obj_views_t = tf.gather(self._obj_views_t, validation_indices_t)
            validation_predictions_t = self._getBatchPredictedData(validation_obj_views_t)

        self._train_full = ForwardModel(ndiffs=n_train_diffs,
                                       indices_t=train_indices_t,
                                       mods_t=train_mods_t,
                                       obj_views_t=train_obj_views_t)
        self._validation = ForwardModel(ndiffs=n_validation_diffs,
                                       indices_t=validation_indices_t,
                                       mods_t=validation_mods_t,
                                       obj_views_t=validation_obj_views_t,
                                       predictions_t=validation_predictions_t)

    def _createBatches(self, batch_size: int) -> None:
        """Use TensorFlow Datasets to create minibatches.
        
        When the diffraction data set is small enough to easily fit in the GPU memory, we can use the minibatch strategy detailed here to avoid I/O bottlenecks. For larger datasets, we have to adopt a slightly different minibatching strategy. More information about minibatches vs timing will be added later on in jupyter notebooks.
        
        In the scenario that the dataset fits into the GPU memory (which we shall assume as a given from now on), we can adopt the strategy: 
        
            1) pre-calculate which (subset of) object pixels the probe interacts with at each scan position. We call these ``obj_views``. Ensure that the order of stacking of these ``obj_views`` match with the order of stacking of the diffraction patterns. 
            
            2) create a list :math:`[0,1,...,N-1]` where :math:`N` is the number of diffraction patterns. Randomly select minibatches from this list (without replacement), then use the corresponding ``obj_view`` and diffraction intensity for further calculation. 
            
            3) Use the iterator framework from TensorFlow to generate these minibatches. Inconveniently, when we use iterators, the minbatches of ``obj_views`` and diffraction patterns thus generated are not stored in the memory---every time we access the iterator, we get a new minibatch. In other words, there is no temporary storage to store this intermediate information at every step. If we want to do finer analysis on the minibatches, we might want this information. 
            For this temporary storage, we can use a TensorFlow Variable object, and store the minibatch information in the variable using an assign operation. The values of TensorFlow variables change only when we use these assign operations. In effect, we only access the iterator when we assign the value to the variable. Otherwise, the value of the variable remains in memory, unconnected to the iterator. Thus the minibatch information is preserved until we use the assign operation again.
        
        After generating a minibatch of ``obj_views``, we use the forward model to generate the predicted diffraction patterns for the current object and probe guesses.
        
        Parameters
        ----------
        batch_size : int
            Number of diffraction patterns in each minibatch.
        """
        
        nd = self._train_full.ndiffs
        batch_size = nd if batch_size==0 else batch_size
        self.batch_size = batch_size
        
        with self._graph.as_default():
            dataset_indices = tf.data.Dataset.from_tensor_slices(self._train_full.indices_t)
            dataset_indices = dataset_indices.apply(tf.data.experimental.shuffle_and_repeat(nd))
            dataset_batch = dataset_indices.batch(batch_size, drop_remainder=True)
            dataset_batch = dataset_batch.apply(tf.data.experimental.prefetch_to_device('/gpu:0', buffer_size=5))
            
            iterator = dataset_batch.make_one_shot_iterator()

            batchi_t = iterator.get_next()
            batch_indices_v = tf.Variable(tf.zeros(batch_size, dtype=tf.int64),
                                        name='batch_indices', trainable=False)
            new_batch_op = batch_indices_v.assign(batchi_t)
            
            batch_mods_t = tf.gather(self._mods_t, batch_indices_v)
            batch_obj_views_t = tf.gather(self._obj_views_t, batch_indices_v)
            batch_predictions_t = self._getBatchPredictedData(batch_obj_views_t)
            
        self._batch_model = ForwardModel(ndiffs=batch_size,
                                   indices_t=batch_indices_v,
                                   mods_t=batch_mods_t,
                                   obj_views_t=batch_obj_views_t,
                                   predictions_t=batch_predictions_t)
        
        self._dataset_indices = dataset_indices
        self._dataset_batch = dataset_batch
        self._new_batch_op = new_batch_op
    
    def _getObjViewsStack(self) -> tf.Tensor:
        """Precalculate the object positioning for each scan position. 
        
        Assumes a small object that is translated within the dimensions of a full-field probe. For each scan position, we translate the object, then pad the object array to the size of the probe beam. For the padding, we assume free-space (transparent) propagation and use 1.0.
        
        In Tensorflow, performing the pad-and-stack procedure in the GPU for complex -valued arrays seems to be buggy. As a workaround, we separately pad-and-stack the real and imaginary parts of the object with 1.0 and 0 respectively.
        
        Returns
        ----------
        obj_views : tensor(complex)
            Stack of tensors that correspond to the padded object at each object translation.
        """
        obj_real_pads = []
        obj_imag_pads = []
        n1 = self.inits.obj_params.obj_w_border_npix
        n2 = self.inits.probe_params.npix
        for p in self.inits.positions:
    
            padded_real = tf.pad(tf.real(self._obj_w_border_t), [[p[0], n2 - (n1+p[0])],
                                                                 [p[1], n2 - (n1+p[1])]],
                                 constant_values=1)
            padded_imag = tf.pad(tf.imag(self._obj_w_border_t), [[p[0], n2 - (n1+p[0])],
                                                                 [p[1], n2 - (n1+p[1])]],
                                 constant_values=0)
            obj_real_pads.append(padded_real)
            obj_imag_pads.append(padded_imag)

        obj_real_pads_t = tf.stack(obj_real_pads)
        obj_imag_pads_t = tf.stack(obj_imag_pads)
        
        obj_views_t = tf.complex(obj_real_pads_t, obj_imag_pads_t)
        return obj_views_t
    
    
    def _getBatchPredictedData(self, obj_views_t: tf.Tensor) -> tf.Tensor:
        """Use the physical forward model to calculate the diffraction patterns for supplied object translations. 
        
        Parameters
        ----------
        obj_views_t : tensor(complex)
            Stack of 2d tensors that each correspond to the object transmission function (including the free space surrounding it) at some scan position.
            
        Returns
        -------
        guess_mods_t : tensor(float32)
            Predicted diffraction patterns for each scan position in ``obj_views``.
        """
        if obj_views_t.get_shape()[0] == 0:
            return tf.zeros(shape=[], dtype='float32')
        
        exit_waves_t = obj_views_t * self._probe_cmplx_t
        out_wavefronts_t = (tf.ifft2d(tf.fft2d(exit_waves_t) * self._prop_kernel_t))
        guess_mods_t = tf.abs(out_wavefronts_t)
        return guess_mods_t
    
    def _getBatchAmplitudeLoss(self, 
                               predicted_data_t: tf.Tensor, 
                               measured_data_t: tf.Tensor) -> tf.Tensor:
        """Get the amplitude (gaussian) loss function for a minibatch. 
        
        The is the amplitude loss, or the loss function for the gaussian noise model. It is a least squares function defined as ``1/2 * sum((predicted_data - measured_data)**2)``.
        
        Parameters
        ----------
        predicted_data_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses the current values of the object and probe variables.
        measured_data_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.
            
        Returns
        -------
        loss_t : tensor(float)
            Scalar value for the current loss function.
        """
        loss_t = 0.5 * tf.reduce_sum((predicted_data_t - measured_data_t)**2)
        return loss_t

    def initLossAndOptimizers(self, 
                             obj_learning_rate: float = 1e-2, 
                             probe_learning_rate: float = 1e1) -> None:
        """
        Set up the training and validation loss functions and the probe and object optimization procedure.
        
        For the training data, we use the loss value for the current minibatch of scan positions. We then try to minimize this loss value by using the Adam optimizer for the object (and probe) variables.
        
        For the validation data, we use the entire validation data set to calculate the loss value. We do not use this for the gradient calculations.
        
        For now, we use the amplitude loss function.
        
        Parameters
        ----------
        obj_learning_rate : float
            Learning rate (or initial step size) for the Adam optimizer for the object variable. Defaults to 0.01.
        probe_learning_rate : float
            Learning rate (or initial step size) for the Adam optimizer for the probe variable. Only applies we enable probe reconstruction. Defaults to 10 
        """
        with self._graph.as_default():
            batch_loss_t = self._getBatchAmplitudeLoss(self._batch_model.predictions_t,
                                                    self._batch_model.mods_t)
            
            validation_loss_t = self._getBatchAmplitudeLoss(self._validation.predictions_t, 
                                                         self._validation.mods_t)
            
            obj_optimizer = tf.train.AdamOptimizer(obj_learning_rate)
            obj_minimize_op = obj_optimizer.minimize(batch_loss_t,
                                                     var_list=[self._obj_v])
            
            self._recons_ops = ReconsLossAndOptimizers(batch_loss_t=batch_loss_t,
                                                      validation_loss_t=validation_loss_t,
                                                      obj_learning_rate=obj_learning_rate,
                                                      obj_opt=obj_optimizer,
                                                      obj_min_op=obj_minimize_op)

            if self.probe_recons:
                self._recons_ops.probe_learning_rate = probe_learning_rate
                self._recons_ops.probe_opt = tf.train.AdamOptimizer(probe_learning_rate)
                self._recons_ops.probe_min_op = self._recons_ops.probe_opt.minimize(batch_loss_t, 
                                                                        var_list=[self._probe_v])
    
    def initSession(self):
        """Initialize the graph and set up the gradient calculation.
        
        Run after creating optimizers."""
        assert hasattr(self, '_recons_ops'), "Create optimizers before initializing the session."
        with self._graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            self.session.run(self._new_batch_op)
    
    def getObjRegistrationError(self) -> float:
        """Calculate the error in the current guess of the object variable.
        
        If ``obj_true`` is subpplied, we use subpixel registration to calculate the error in the current guess of the object transmision function. Otherwise, the function returns ``NaN``.
        
        Returns
        -------
        err : float
            Subpixel registration error.
        """
        otrue = self.obj_true
        if otrue is None:
            return np.nan
        recons_obj = self.session.run(self._obj_cmplx_t)
        shift, err, phase = register_translation(recons_obj, otrue, upsample_factor=10)
        shift, err, phase = register_translation(recons_obj * np.exp(-1j * phase), otrue, upsample_factor=10)
        return err

    def getProbeRegistrationError(self) -> float:
        """Calculate the error in the current guess of the probe variable.
        
        If ``probe_recons`` is enabled and ``probe_true`` is subpplied, we use subpixel registration to calculate the error in the current guess of the object transmision function. Otherwise, the function returns ``NaN``.
        
        Returns
        -------
        err : float
            Subpixel registration error.
        """
        ptrue = self.probe_true
        if (ptrue is None) or (not self.probe_recons):
            return np.nan
        recons_probe = self.session.run(self._probe_cmplx_t)
        shift, err, phase = register_translation(recons_probe, ptrue, upsample_factor=10)
        shift, err, phase = register_translation(recons_probe * np.exp(-1j * phase), ptrue, upsample_factor=10)
        return err
    
    def updateOutputs(self) -> None:
        """Populate numpy arrays from the current values of the probe and object tensorflow variables.""" 
        obj_out, probe_out = self.session.run([self._obj_cmplx_t, self._probe_cmplx_t])
        self.outs.obj = obj_out
        self.outs.probe = probe_out
    
    def saveOutputs(self, out_name: str)-> Tuple[str]:
        """Save the current probe and object values and the current log to disk.
        
        The files saved are named as::
            
            * log file : <out_name>.pkl
            * object transmission function : <out_name>_obj.npy
            * probe transmission function : <out_name>_probe.npy
        
        Parameters
        ----------
        out_name : str
            Common prefix for the saved output files.
            
        Returns
        -------
        log_pkl, obj_npy, probe_npy : str
            Names of the saved files for the log, the object function, and the probe function respectively.
        """
        log_pkl = f'{out_name}.pkl'
        obj_npy = f'{out_name}_obj.npy'
        probe_npy = f'{out_name}_probe.npy'
        self.outs.log.to_pickle(log_pkl)
        np.save(obj_npy, self.outs.obj)
        np.save(probe_npy, self.outs.probe)
        return log_pkl, obj_npy, probe_npy
        
    def run(self,
            validation_frequency: int = 1,
            improvement_threshold: float = 5e-4,
            patience: int = 50,
            patience_increase_factor: float = 1.5,
            max_iters: int = 5000,
            debug_output: bool = True,
            debug_output_frequency: int = 10,
            probe_fixed_epochs: int=0) -> None:
        """Perform the optimization a specified number of times (with early stopping).
        
        This command provides fine control over the number of times we run the minization procedure and over the early stopping criterion. To understand how this works, we first introduce some terminology:
            - ``iteration``: Every minibatch update counts as one iteration.
            - ``epoch``: A single pass through the entire data set counts as one epoch. In the minibatch setting, each epoch usually consists of multiple iterations.
            - ``patience``: When there is no improvement in the minimum loss value obtained after an epoch of optimization, we can either pull the trigger immediately, or wait for a fixed number of epochs (without improvement) before pulling the trigger. This fixed number of epochs where we wait, even when we see no improvement, is the patience.
            - ``patience_increase_factor``: Typically, during the optimization procedure, we expect a fast improvement in the loss value at the beginning of the optimization procedure, with the rate of improvement slowing down as we proceed with the optimization. To account for this, we want an early stopping procedure with low patience at the beginning, and with increasing patience as we move towards the minimum. The `patience_increase_factor` controls the rate of increase of the patience (which depends on the `validation_frequency` parameter).
                
        Parameters
        ----------
        validation_frequency : int
            Number of epochs between each calculation of the validation loss. This is also the number of epochs between each check (and update) of the `patience` parameter. 
        improvement_threshold : float
            Relative tolerance for ``improvement`` of the minimum loss value, where ``improvement`` is defined as ``improvement = abs(validation_best_loss - validation_loss) / validation_best_loss``.
        patience : int
            Minimum allowable number of epochs betweeen improvement in the minimum loss value, where the ``improvement`` is as defined by `improvement_threshold`. The `patience` is increased dynamically (depending on the `patience_increase_factor` and the `validation_frequency`)  during the optimization procedure.
        patience_increase_factor : float
            Factor by which the patience is increased whenever the ``improvement`` is better than the `improvement_threshold`. 
        max_iters : int
            Maximum number of ``iterations`` to perform. Each ``epoch`` is usually composed of multiple iterations.
        debug_output : bool
            Whether to print the validation log output to the screen.
        debug_output_frequency : int
            Number of validation updates after which we print the validation log output to the screen.
        probe_fixed_epochs : int
            Number of epochs (at the beginning) where we only adjust the object variable.
        """
        
        if debug_output:
            print(f"{'iter':<8}{'epoch':<7}{'train_loss':<12}{'obj_err':<10}{'probe_err':<10}" +
                  f"{'patience':<10}{'val_loss':<10}{'val_best_loss':<13}")
        
        epochs_this = 0
        log = self.outs.log
        index = len(log)
        for i in range(max_iters):
            ix = index + i
            
            if self.probe_recons and epochs_this >= probe_fixed_epochs: 
                _ = self.session.run(self._recons_ops.probe_min_op)
            
            lossval, _ = self.session.run([self._recons_ops.batch_loss_t, 
                                           self._recons_ops.obj_min_op])
            _ = self.session.run(self._new_batch_op)
            log.loc[ix, 'loss'] = lossval
            
            if ix == 0:
                log.loc[0, 'epoch'] = 0
                continue
            elif ix % (self._train_full.ndiffs // self.batch_size) != 0:
                log.loc[ix, 'epoch'] = log['epoch'][ix-1]
                continue
            
            log.loc[ix, 'epoch'] = log['epoch'][ix-1] + 1 
            epochs_this += 1
            
            if epochs_this % validation_frequency != 0:
                continue
            validation_lossval = self.session.run(self._recons_ops.validation_loss_t)
            log.loc[ix, 'validation_loss'] = validation_lossval
            
            obj_registration_error = self.getObjRegistrationError()
            log.loc[ix, 'obj_error'] = obj_registration_error
            
            probe_registration_error = self.getProbeRegistrationError()
            log.loc[ix, 'probe_error'] = probe_registration_error
            
            validation_best_loss = np.inf if ix == 0 else log['validation_loss'][:-1].min()
            
            if validation_lossval <= validation_best_loss:
                if np.abs(validation_lossval - validation_best_loss) > validation_best_loss * improvement_threshold:
                    patience = max(patience, epochs_this * patience_increase_factor)
                
            log.loc[ix, 'patience'] = patience
                
            if debug_output and epochs_this % (debug_output_frequency * validation_frequency)== 0:
                print(f'{i:<8} '
                       +f'{epochs_this:<7}'
                       + f'{lossval:<12.7g} '
                       + f'{obj_registration_error:<10.7g} '
                       + f'{probe_registration_error:<10.7g} '
                       + f'{patience:<10.7g} '
                       + f'{validation_lossval:<10.7g} '
                       + f'{validation_best_loss:<13.7g}')
            
            if epochs_this >= patience:
                break
        self.updateOutputs()
    
    def genPlotsRecons(self) -> None:
        """Plot the reconstructed probe and object amplitudes and phases."""
        self.updateOutputs()
        
        plt.figure(figsize=[14,3])
        plt.subplot(1,4,1)
        plt.pcolormesh(np.abs(self.outs.obj), cmap='gray')
        plt.colorbar()
        plt.subplot(1,4,2)
        plt.pcolormesh(np.angle(self.outs.obj), cmap='gray')
        plt.subplot(1,4,3)
        plt.pcolormesh(np.abs(self.outs.probe), cmap='gray')
        plt.colorbar()
        plt.subplot(1,4,4)
        plt.pcolormesh(np.angle(self.outs.probe), cmap='gray')
        plt.colorbar()
        plt.show()
    
    def genPlotMetrics(self) -> None:
        """Plot the metrics recorded in the log."""
        log = self.outs.log
        fig, axs = plt.subplots(1,4,figsize=[14,3])
        axs[0].plot(np.log(log['loss'].dropna()))
        axs[0].set_title('loss')
        
        axs[1].plot(log['obj_error'].dropna())
        axs[1].set_title('obj_error')
        
        axs[2].plot(log['probe_error'].dropna())
        axs[2].set_title('probe_error')
        
        axs[3].plot(np.log(log['validation_loss'].dropna()))
        axs[3].set_title('validation_loss')
        plt.show()
        
    def _getClipOp(self, max_abs: float=1.0) -> None:
        """Not used for now"""
        with self.graph.as_default():
            obj_reshaped = tf.reshape(self.tf_obj, [2, -1])
            obj_clipped = tf.clip_by_norm(obj_reshaped, max_abs, axes=[0])
            obj_clipped_reshaped = tf.reshape(obj_clipped, [-1])
            clipped = tf.assign(self.tf_obj, obj_clipped_reshaped, name='clip_op')
        return clipped



class tfNFPtychoReconsFromSimulation(tfNFPtychoRecons):
    
    def __init__(self,
                 simulation: NFPtychoSimulation,
                 obj_guess: Optional[np.ndarray] = None,
                 probe_guess: Optional[np.ndarray] = None,
                 probe_recons: bool = True,
                 batch_size: int = 0,
                 n_validation_diffs: int = 0) -> None:
        
        """
        """
        self._simulation = simulation
        obj_args = self._checkGuessNpix(dt.asdict(simulation._obj_params), obj_guess)
        probe_args = self._checkGuessNpix(dt.asdict(simulation._probe_params), probe_guess)
        probe_params = self._setParams(ProbeParams, probe_args)
        obj_params = self._setParams(ObjParams, obj_args)
        det_params = self._setParams(DetectorParams, dt.asdict(simulation._det_params))
        
        self.probe_recons = probe_recons
        if probe_recons == False and probe_guess is None:
            probe_guess = simulation.probe
        
        self.inits = ReconsInits(positions=simulation.positions,
                                 mods=simulation.diffraction_mods,
                                 obj_params=obj_params, 
                                 probe_params=probe_params,
                                 det_params=det_params,
                                 obj=obj_guess,
                                 probe=probe_guess)
        
        
        
        # Tensorflow setup
        self._createTFModels(n_validation_diffs)
        self._createBatches(batch_size)
        
        self.obj_true = simulation.obj
        self.probe_true = simulation.probe
        
        log = DataFrame(columns=['loss','epoch','obj_error','probe_error','validation_loss','patience'],
                              dtype='float32')
        self.outs = ReconsOutputs(np.zeros_like(obj_guess),
                                  np.zeros_like(probe_guess),
                                  log)
        

