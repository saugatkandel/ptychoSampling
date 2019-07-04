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
from dataclasses import dataclass



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
    stdev : :obj:`float`, optional
        Standard deviation of the gaussian. The function requires either the standard deviation or the fwhm. 
    fwhm : :obj:`float`, optional
        Full width at half maximum (FWHM) of the peak. The function requires either the standard deviation or the fwhm. If we supply the fwhm, the standard deviation is calculated as :math:`stdev = fwhm / 2.35682`.
    
    Returns
    -------
    out: float ndarray
        A 2d array of shape npix X npix and dtype float32.
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



def getTFPropKernel(beam_shape: tuple, 
                    pixel_pitch: float, 
                    wavelength: float, 
                    prop_dist: float,
                    fftshift: bool = True) -> np.ndarray:
    r"""Generates a kernel for wavefield propagation using the Transfer function method. 
    
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
        Whether to return the kernel after performing an fftshift. Since we most often use the output kernel (H) inside the structure IFFT(FFT(psi) * H) where psi is the wavefront that we need to propagate, it is convenient to deal with the shifted kernel directly. 
    
    Returns
    --------
    out : complex ndarray
        A 2d array of shape beam_shape and dtype complex64.
    
    References
    ----------
    .. [2] Chapter 5.1 of the book "Computational Fourier Optics: A MATLAB Tutorial" by David Voelz (2011).
    .. [3] Saugat Kandel, S. Maddali, Marc Allain, Stephan O. Hruszkewycz, Chris Jacobsen, and Youssef S. G. Nashed, "Using automatic differentiation as a general framework for ptychographic reconstruction," Opt. Express 27, 18653-18672 (2019)
    .. [4] Richard M. Clare, Marco Stockmar, Martin Dierolf, Irene Zanette, and Franz Pfeiffer, "Characterization of near-field ptychography," Opt. Express 23, 19728-19742 (2015)
    """
    
    M, N = beam_shape
    
    fx = np.fft.fftfreq(M, d=pixel_pitch)
    fy = np.fft.fftfreq(N, d=pixel_pitch)
    
    FX, FY = np.meshgrid(fx, fy)
    FX = np.fft.fftshift(FX)
    FY = np.fft.fftshift(FY)
    
    H = np.exp(-1j * np.pi * wavelength * prop_dist * (FX**2 + FY**2))
    if fftshift: 
        H = np.fft.fftshift(H)
    return H.astype('complex64')



def getSpeckle(npix: int, 
               window_size: int) -> np.ndarray:
    """Generates a speckle pattern. 
    
    To generate a speckle pattern, this function uses a window_size x window_size array of complex numbers with unit amplitude and uniformly random phase. This array is padded with zeros to get an npix x npix array, an FFT of which gives us a speckle pattern. The speckle pattern thus generated is discontinuous; there is a phase step of pi between adjacent pixels in both the x and y directions. We remove these discontinuities to get the final, continuous, speckle pattern.
    
    Parameters
    ----------
    npix : int
        Number of pixels along each side of the 2d array containing the speckle pattern.
    window_size : int 
        The size of the rectangular window used to generate the speckle pattern. Larger window sizes give smaller speckle sizes and vice versa. (*Note*: I tried a circular window as well, but the results did not change 
    noticeably.)
    
    Returns
    --------
    out : complex ndarray
        A 2d array of size npix x npix and dtype complex64.
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



@dataclass(frozen=True)
class ObjParams:
    """Convenience class to store the object parameters.
    
    Note
    -----
    Adding an assumed-known border (filled with ones) to the object helps avoid the affine phase ambiguity.
    
    Attributes
    ----------
    obj_npix : int
        Number of pixels in each side of the (square) object to be generated.
    mod_range : float
        Maximum value of the modulus for the object pixels.
    phase_range : float
        Maximum value of the phase for the object pixels. 
    border_npix : int
        Number of pixels to add along the border in the left, right, top, and bottom margins of the object. If :obj:`border_npix = 10`, then the simulation adds 10 pixels to the left of the object, and 10 pixels to the right of the object, i.e. a total of 20 pixels along the x-direction (and similarly with y). 
    border_const : float
        Constant value to fill the border with.
    """
    obj_npix: int
    mod_range: float
    phase_range: float
    border_npix: int
    border_const: float
    
    @property
    def obj_w_border_npix(self) -> int:
        """Total number of pixels (including object and border) in each side of the object."""
        return self.obj_npix + 2 * self.border_npix

@dataclass(frozen=True)
class ProbeParams:
    """Convenience class to store the probe parameters.

    Attributes
    ----------
    wavelength : float
        Wavelength of the probe beam.
    npix : int
        Number of pixels in each side of the square probe to be generated.
    photons_flux : float 
        Average number of photons per pixel in the probe beam. 
    """
    wavelength: float
    npix: int
    photons_flux: float
    
    @property
    def n_photons(self) -> float:
        """Total number of photons in the beam."""
        return self.photons_flux * self.npix**2

@dataclass(frozen=True)
class DetectorParams:
    """Convenience class to store the detector parameters.
    
    Attributes
    -----------
    obj_dist : float
        Object-detector distance (in m).
    pixel_pitch : float
        Width of each individual pixel (in m). 
    """
    obj_dist: float
    pixel_pitch: float

@dataclass(frozen=True)
class ScanParams:
    """Convenience class to store the ptychographic scan parameters.

    Attributes
    ----------
    scan_step_npix : int
        Number of pixels per step in the raster grid.
    poisson_noise : bool
        Whether to simulate Poisson noise in the diffraction data.
    """
    scan_step_npix: int
    poisson_noise: bool



@dataclass(frozen=True)
class NFSimObjParams(ObjParams): 
    """Customizing ObjParams with some default parameters for nearfield simulations.
    
    Only function is to use the default parameters. Does not extend ObjParams. 
    
    See ObjParams for documentation on the individual parameters used.
    """
    obj_npix : int = 192
    mod_range: float = 1.0
    phase_range: float = np.pi
    border_npix: int = 32
    border_const: float = 1.0

@dataclass(frozen=True)
class NFSimDetectorParams(DetectorParams):
    """Customizing DetectorParams with some default parameters for nearfield simulations.
    
    Only function is to use the default parameters. Does not extend ObjParams. 
    
    See ObjParams for documentation on the individual parameters used.."""
    obj_dist: float = 0.0468
    pixel_pitch: float = 3e-7

@dataclass(frozen=True)
class NFSimProbeParams(ProbeParams):
    """Customizes ProbeParams with default parameter values and also adds parameters specific to the nearfield simulation.
    
    For detail on the other attributes and parameters, refer to the documentation for the ProbeParams class.
        
    Attributes
    ----------
    gaussian_intensity_stdev_pix : float
        Standard deviation of the gaussian probe to be used. See the function getGaussian2D for more detailed information on this parameter.
    speckle_window_pix : int
        The size of the rectangular window used to generate the speckle pattern. See the function getSpeckle for more detail on this parameter. 
    """
    wavelength: float = 0.142e-9
    npix: int = 512
    photons_flux: float = 1e4
    gaussian_intensity_stdev_npix: float = 150.0
    speckle_window_npix: int = 40

@dataclass(frozen=True)
class NFSimScanParams(ScanParams):
    r""""Customizes ScanParams with default parameter values and also adds parameters specific to the nearfield simulation.
    
    For detail on the other parameters, refer to the documentation for the ScanParams class.
        
    Attributes
    ----------
    scan_area_buffer_npix : int
        This is a bit of hack to ensure that the scan area is around the center of the full-field probe. For e.g., if scan_area_buffer_npix = 20, we start the ptychographic scan from the coordinate (20, 20) of the probe instead of from (0,0). The regions between x=[0,20] and y=[0,20] are never sampled.
        Basically, this is like imposing a margin of 20 pixels in the left and bottom of the probe.
        
        """
    scan_step_npix: int = 44
    poisson_noise: bool = True
    scan_area_buffer_npix: int = 20



class NFPtychoSimulation(object):
    r"""Simulate a near-field ptychography simulation using a full-field probe and object translations.
        
    I don't know if this is the typical scenario for a near-field ptychography experiment. For this work, I am trying to use a similar setup to that used by Clare et al in [6]_.

    See NFSimObjParams, NFSimProbeParams, NFSimDetectorParams, and NFSimScanParams classes for details on the options available to customize the simulation.
    
    Attributes
    ----------
    obj_params : NFSimObjParams
        Parameters for the simulated object.
    probe_params : NFSimProbeParams
        Parameters for simulated probe.
    det_params : NFSimDetectorParams
        Parameters for simulated detector.
    scan_params : NFSimScanParams
        Parameters for the scan grid and noise setup.
    obj_true: complex ndarray
        Holds the generated complex valued object.
    obj_w_border: complex ndarray
        Holds the generated complex valued object along with the border specified through obj_params.
    probe_true: complex ndarray
        Holds the generated complex valued probe.
    prop_kernel: complex ndarray
        Holds the (fft-shifted) Fresnel propagation kernel for the simulation parameters. 
    positions : int ndarray
        Holds the scan positions (object translations) used for the ptychographic scan.
    diffraction_mods : float ndarray
        Holds the modulus of the exit wave at the detector plane for the scan positions. This is just the square root of the intensity pattern at the object plane.
    
    Parameters
    -----------
    obj_args : dict
        Dictionary that contains pairs of arguments and values to use to create a custom NFSimObjParams class, and thus change the parameters of the simulated object. 
    probe_args : dict
        Dictionary that contains pairs of arguments and values to use to create a custom NFSimProbeParams class, and thus change the parameters of the simulated probe. 
    detector_args : dict
        Dictionary that contains pairs of arguments and values to use to create a custom NFSimDetectorParams class, and thus change the parameters of the detector used. 
    scan_args : dict
        Dictionary that contains pairs of arguments and values to use to create a custom NFSimScanParams class, and thus change the parameters of the ptychographic scan. 

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
        
        """
    def __init__(self,
                 obj_args: dict = {},
                 probe_args: dict = {},
                 detector_args: dict = {},
                 scan_args: dict = {}) -> None:
  
        self.obj_params = NFSimObjParams(**obj_args)
        self.probe_params = NFSimProbeParams(**probe_args)
        self.det_params = NFSimDetectorParams(**detector_args)
        self.scan_params = NFSimScanParams(**scan_args)
        
        self.checkValidity()
        
        # Generating the simulated object
        self.obj_true = getSampleObj(npix=self.obj_params.obj_npix,
                                      mod_range=self.obj_params.mod_range,
                                      phase_range=self.obj_params.phase_range)
        pad = self.obj_params.border_npix
        self.obj_w_border = np.pad(self.obj_true, [[pad,pad],[pad,pad]], 
                                      mode='constant',
                                      constant_values=self.obj_params.border_const)
        
        # Generating the simulated probe
        gaussian_intensity = getGaussian2D(self.probe_params.npix, 
                                           self.probe_params.gaussian_intensity_stdev_npix)
        gaussian_ampl = gaussian_intensity**0.5
        speckle = getSpeckle(self.probe_params.npix, 
                             self.probe_params.speckle_window_npix)
        probe_data = gaussian_ampl * speckle
        self.probe_true = probe_data * np.sqrt(self.probe_params.n_photons 
                                         / np.sum(np.abs(probe_data)**2))
        
        # Generating the fft-shifted propagation kernel 
        self.prop_kernel = getTFPropKernel(beam_shape=self.probe_true.shape,
                                           pixel_pitch=self.det_params.pixel_pitch,
                                           wavelength=self.probe_params.wavelength,
                                           prop_dist=self.det_params.obj_dist)
        
        self.genPtychographyPositions()
        self.genDiffractionMods()
        self.ndiffs = self.diffraction_mods.shape[0]    
        
    def checkValidity(self) -> None:
        """Checking if the parameters supplied are valid. 
        
        For now, we only check to ensure that the step size supplied is larger than the 
        width of the Fresnel zone for the simulation parameters. This ensures that the 
        generated diffraction patterns have enough diversity.
        
        Need to add more checks here.
        """
        
        fresnel_zone_dist = np.sqrt(self.probe_params.wavelength * self.det_params.obj_dist)
        fresnel_zone_npix = fresnel_zone_dist / self.det_params.pixel_pitch
        
        error_str = (f"Step size ({self.scan_params.scan_step_npix} is too small. "
                     + f"Ensure that the step size is at least larger than the Fresnel zone width "
                     + f"({fresnel_zone_npix}) to ensure diversity in the diffraction patterns.")
        assert self.scan_params.scan_step_npix > fresnel_zone_npix, error_str  
    
    def genPtychographyPositions(self) -> None:
        """Generate the scan positions for the ptychographic scan."""
        
        p1 = self.scan_params.scan_area_buffer_npix
        p2 = self.probe_params.npix - p1 - self.obj_params.obj_w_border_npix
        positions_x = np.arange(p1, p2, self.scan_params.scan_step_npix)
        positions = []
        
        for r in positions_x:
            for c in positions_x:
                positions.append([r,c])
        self.positions = np.array(positions)
    
    def genDiffractionMods(self) -> None:
        """Generate the near-field diffraction patterns for the ptychography scan using the transfer function method."""
        diffraction_intensities = []
        
        npix_pad = self.probe_params.npix - self.obj_params.obj_w_border_npix
        obj_padded_to_probe = np.pad(self.obj_w_border, 
                                     [[0, npix_pad], [0, npix_pad]],
                                     mode='constant',
                                     constant_values=1.0)
        for indx, (r,c) in enumerate(self.positions):
            exit_wave = self.probe_true * np.roll(obj_padded_to_probe, [r,c], axis=(0,1)) 
            nearfield_wave = np.fft.ifftshift(np.fft.ifft2(self.prop_kernel * np.fft.fft2(exit_wave)))
            diffraction_intensities.append(np.abs(nearfield_wave)**2)
            
        if self.scan_params.poisson_noise: 
            diffraction_intensities = np.random.poisson(diffraction_intensities)
        self.diffraction_mods = np.sqrt(diffraction_intensities)
        
    def getScatterIndices(self) -> None:
        """Not in use right now"""
        scatter_indices_all = []
        for py, px in self.positions:
            R, C = np.ogrid[py:self.obj_params.obj_w_border_npix + py, 
                            px:self.obj_params.obj_w_border_npix + px]
            scatter_single = ((R % self.probe_params.npix) * self.probe_params.npix + 
                              (C % self.probe_params.npix))
            scatter_indices_all.append(scatter_single)
        scatter_indices =  np.array(scatter_indices_all)



class tfNearFieldPtychoRecons(object):
    """Reconstruct the object and probe from a near-field ptychography simulation. 

    Assumes that the probe is full-field, the object is (much) smaller than the probe, and the object is translated within the probe field to generate the diffraction patterns. 

    Assumes square object and probe.

    Uses the gaussian noise model for the loss function. This is easy to change if neeeded.

    Attributes
    ----------
    test
    """
    
    def __init__(self,
                 positions: np.ndarray,
                 diffraction_mods: np.ndarray,
                 wavelength: float,
                 obj_detector_dist: float,
                 detector_pixel_pitch: float,
                 obj_npix: int = 0,
                 obj_border_npix: int = 0,
                 obj_border_const: float = 0.0,
                 obj_mod_range: float = 1.0,
                 probe_npix: int = 0,
                 obj_guess: Optional[np.ndarray] = None,
                 probe_guess: Optional[np.ndarray] = None,
                 probe_recons: bool = False,
                 batch_size: int = 0,
                 n_validation_diffs: int = 0,
                 obj_true: Optional[np.ndarray] = None,
                 probe_true: Optional[np.ndarray] = None) -> None:
                 

        
        self.positions = positions
        self.diffraction_mods = diffraction_mods
        self.diffraction_mods_shifted = np.fft.fftshift(self.diffraction_mods, axes=(-1,-2))
        
        obj_npix = obj_guess.shape[0] if obj_guess is not None else obj_npix
        assert obj_npix > 0, "Need to supply either obj_guess or obj_npix"
        self.obj_params = ObjParams(obj_npix=obj_npix, 
                                    mod_range=obj_mod_range,
                                    border_npix=obj_border_npix,
                                    border_const=obj_border_const)
        
        self.probe_recons = probe_recons
        probe_npix = probe_guess.shape[0] if probe_guess is not None else probe_npix
        assert probe_npix > 0, "Need to supply either probe_guess or probe_npix"
        if not probe_recons:
            assert (probe_guess is not None), "Need to supply probe array if probe_recons is False."
        
        self.probe_params = ProbeParams(npix=probe_npix,
                                        wavelength=wavelength)
        
        self.det_params = DetectorParams(obj_dist=obj_detector_dist,
                                         pixel_pitch=detector_pixel_pitch)
        
        self.prop_kernel = getTFPropKernel(beam_shape=(probe_npix, probe_npix),
                                           pixel_pitch=detector_pixel_pitch,
                                           wavelength=wavelength,
                                           prop_dist=obj_detector_dist)
        
        self.setObjProbeGuess(obj_guess, probe_guess)
        self.setTrainingAndValidation(n_validation_diffs, batch_size)
        
        # Tensorflow setup
        self.createGraphAndVars()
        self.initDataSet()
        
        self.obj_true = obj_true
        self.probe_true = probe_true
        self.optimizers_defined = False
        
        self.data = DataFrame(columns=['loss','epoch','obj_error','probe_error','validation_loss','patience'],
                              dtype='float32')
        
    
    def setTrainingAndValidation(self, 
                                 n_validation_diffs,
                                 batch_size):
        self.ndiffs = self.diffraction_mods.shape[0]
        
        self.n_validation_diffs = n_validation_diffs
        self.validation_indices = np.random.permutation(self.ndiffs)[:self.n_validation_diffs]
        
        self.train_ndiffs = self.ndiffs - self.n_validation_diffs
        self.train_indices = np.array([i for i in range(self.ndiffs) if i not in self.validation_indices])
        
        self.batch_size = self.train_ndiffs if batch_size==0 else batch_size 
    
    def setObjProbeGuess(self, obj_guess, probe_guess):
        
        if obj_guess is None:
            n = self.obj_params.obj_npix
            obj_guess = (np.random.random((n,n)) * np.exp(1j * np.random.random((n,n)) * np.pi))
        self.obj_guess = obj_guess 

        if probe_guess is None:
            mods_avg = np.mean(self.diffraction_mods_shifted, axis=0)
            probe_guess = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(mods_avg) 
                                                        / self.prop_kernel))
        self.probe_guess = probe_guess
        
        
    def createGraphAndVars(self) -> None:
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_obj_var = tf.Variable(np.array([np.real(self.obj_guess), 
                                                    np.imag(self.obj_guess)]),
                                          dtype='float32')
            self.tf_obj = tf.complex(self.tf_obj_var[0], self.tf_obj_var[1])
            
            
            self.tf_probe_var = tf.Variable(np.array([np.real(self.probe_guess), 
                                                    np.imag(self.probe_guess)]),
                                          dtype='float32')
            self.tf_probe = tf.complex(self.tf_probe_var[0], self.tf_probe_var[1])

            self.tf_train_mods = tf.constant(self.diffraction_mods_shifted[self.train_indices],
                                                        dtype='float32')
            self.tf_validation_mods = tf.constant(self.diffraction_mods_shifted[self.validation_indices],
                                                             dtype='float32')

            self.tf_prop_kernel = tf.constant(self.prop_kernel, dtype='complex64', name='propagation_kernel')
            
            pad = self.obj_params.border_npix
            self.tf_obj_w_border = tf.pad(self.tf_obj, [[pad, pad], [pad, pad]], 
                                          constant_values=self.obj_params.border_const)
            
            self.tf_train_obj_views = self.getObjViewsStack(self.train_indices)
            self.tf_validation_obj_views = self.getObjViewsStack(self.validation_indices)
    
    def initDataSet(self) -> None:
        with self.graph.as_default():
            dataset_indices = tf.data.Dataset.range(self.train_ndiffs)
            dataset_indices = dataset_indices.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.train_ndiffs))
            dataset_batch = dataset_indices.batch(self.batch_size, drop_remainder=True)
            self.dataset_batch = dataset_batch.apply(tf.data.experimental.prefetch_to_device('/gpu:0', buffer_size=5))
            
            self.iterator = self.dataset_batch.make_one_shot_iterator()

            batchi = self.iterator.get_next()
            self.batch_indices = tf.Variable(tf.zeros(self.batch_size, dtype=tf.int64),
                                             name='batch_indices', trainable=False)
            self.batch_assign_op = self.batch_indices.assign(batchi)
            
            self.batch_train_mods = tf.gather(self.tf_train_mods, self.batch_indices)
            self.batch_train_obj_views = tf.gather(self.tf_train_obj_views, self.batch_indices)
    
    def getObjViewsStack(self, indices):
        obj_real_pads = []
        obj_imag_pads = []
        n1 = self.obj_params.obj_w_border_npix
        n2 = self.probe_params.npix
        for p in self.positions[indices]:
    
            padded_real = tf.pad(tf.real(self.tf_obj_w_border), [[p[0], n2 - (n1+p[0])],
                                                                 [p[1], n2 - (n1+p[1])]],
                                 constant_values=1)
            padded_imag = tf.pad(tf.imag(self.tf_obj_w_border), [[p[0], n2 - (n1+p[0])],
                                                                 [p[1], n2 - (n1+p[1])]],
                                 constant_values=0)
            obj_real_pads.append(padded_real)
            obj_imag_pads.append(padded_imag)

        obj_real_pads = tf.stack(obj_real_pads)
        obj_imag_pads = tf.stack(obj_imag_pads)
        
        obj_views = tf.complex(obj_real_pads, obj_imag_pads)
        return obj_views
    
    
    def getBatchPredictedData(self, obj_views) -> tf.Tensor:
        if obj_views.get_shape()[0] == 0:
            return tf.zeros(shape=[], dtype='float32')
        
        exit_waves = obj_views * self.tf_probe
        out_wavefronts = (tf.ifft2d(tf.fft2d(exit_waves) * self.tf_prop_kernel))
        guess_mods = tf.abs(out_wavefronts)
        return guess_mods
        

    def getBatchAmplitudeLoss(self, predicted_data, measured_data) -> tf.Tensor:
        loss = 0.5 * tf.reduce_sum((predicted_data - measured_data)**2)
        return loss

    def setLossAndOptimizers(self, 
                             obj_learning_rate: float = 1e-2, 
                             probe_learning_rate: float = 1e-1) -> None:
        with self.graph.as_default():
            self.training_predictions = self.getBatchPredictedData(self.batch_train_obj_views)
            self.training_loss = self.getBatchAmplitudeLoss(self.training_predictions,
                                                            self.batch_train_mods)
            
            self.validation_predictions = self.getBatchPredictedData(self.tf_validation_obj_views)
            self.validation_loss = self.getBatchAmplitudeLoss(self.validation_predictions, 
                                                              self.tf_validation_mods)
            
            self.obj_learning_rate = obj_learning_rate
            self.probe_learning_rate = probe_learning_rate
            
            self.obj_optimizer = tf.train.AdamOptimizer(self.obj_learning_rate)
            self.obj_minimize_op = self.obj_optimizer.minimize(self.training_loss,
                                                               var_list=[self.tf_obj_var])
            
            if self.probe_recons:
                self.probe_optimizer = tf.train.AdamOptimizer(self.probe_learning_rate)
                self.probe_minimize_op = self.probe_optimizer.minimize(self.training_loss, 
                                                                       var_list=[self.tf_probe_var])
        self.optimizers_defined = True
    
    def initSession(self):
        assert self.optimizers_defined, "Create optimizers before initializing the session."
        with self.graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            self.session.run(self.batch_assign_op)
            
    def getObjRegistrationError(self):
        if self.obj_true is None:
            return np.nan
        recons_obj = self.session.run(self.tf_obj)
        shift, err, phase = register_translation(recons_obj, self.obj_true, upsample_factor=10)
        shift, err, phase = register_translation(recons_obj * np.exp(-1j * phase), self.obj_true, upsample_factor=10)
        return err

    def getProbeRegistrationError(self):
        if (self.probe_true is None) or (not self.probe_recons):
            return np.nan
        recons_probe = self.session.run(self.tf_probe)
        shift, err, phase = register_translation(recons_probe, self.probe_true, upsample_factor=10)
        shift, err, phase = register_translation(recons_probe * np.exp(-1j * phase), self.probe_true, upsample_factor=10)
        return err
    
    def run(self, 
            validation_frequency: int = 1,
            improvement_threshold: float = 5e-4,
            patience: int = 50,
            patience_increase_factor: float = 1.5,
            max_iters: int = 5000,
            debug_output: bool = True,
            debug_output_epoch_frequency: int = 10,
            probe_fixed_epochs=0) -> None:
        
        print('epochs','training_loss','obj_err','probe_err','patience',
              'validation_loss','validation_best_loss')
        
        epochs_this = 0
        index = len(self.data)
        for i in tqdm(range(max_iters)):
            ix = index + i
            
            if self.probe_recons and epochs_this >= probe_fixed_epochs: 
                _ = self.session.run(self.probe_minimize_op)
            
            lossval, _ = self.session.run([self.training_loss, 
                                           self.obj_minimize_op])
            _ = self.session.run(self.batch_assign_op)
            self.data.loc[ix, 'loss'] = lossval
            
            if ix == 0:
                self.data.loc[0, 'epoch'] = 0
                continue
            elif ix % (self.train_ndiffs // self.batch_size) != 0:
                self.data.loc[ix, 'epoch'] = self.data['epoch'][ix-1]
                continue
            
            self.data.loc[ix, 'epoch'] = self.data['epoch'][ix-1] + 1 
            epochs_this += 1
            
            if epochs_this % validation_frequency != 0:
                continue
            validation_lossval = self.session.run(self.validation_loss)
            self.data.loc[ix, 'validation_loss'] = validation_lossval
            
            obj_registration_error = self.getObjRegistrationError()
            self.data.loc[ix, 'obj_error'] = obj_registration_error
            
            probe_registration_error = self.getProbeRegistrationError()
            self.data.loc[ix, 'probe_error'] = probe_registration_error
            
            validation_best_loss = np.inf if ix == 0 else self.data['validation_loss'][:-1].min()
            
            if validation_lossval <= validation_best_loss:
                if np.abs(validation_lossval - validation_best_loss) > validation_best_loss * improvement_threshold:
                    patience = max(patience, epochs_this * patience_increase_factor)
                
            self.data.loc[ix, 'patience'] = patience
                
            if debug_output and epochs_this % debug_output_epoch_frequency == 0:
                print(f'{epochs_this} '
                       + f'{lossval:8.7g} '
                       + f'{obj_registration_error:8.7g} '
                       + f'{probe_registration_error:8.7g} '
                       + f'{patience:8.7g} '
                       + f'{validation_lossval:8.7g} '
                       + f'{validation_best_loss:8.7g}')
            
            if epochs_this >= patience:
                break
                
    
    def genPlotsRecons(self):
        
        recons_obj = self.session.run(self.tf_obj)[npix:-npix, npix:-npix]
        
        plt.figure(figsize=[8,3])
        plt.subplot(1,2,1)
        plt.pcolormesh(np.abs(recons_obj), cmap='gray')
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.pcolormesh(np.angle(recons_obj), cmap='gray')
        plt.colorbar()
        plt.show()
    
    def genPlotMetrics(self):
        fig, axs = plt.subplots(1,3,figsize=[12,3])
        axs[0].plot(np.log(self.data['loss'].dropna()))
        axs[0].set_title('loss')
        
        axs[1].plot(self.data['obj_error'].dropna())
        axs[1].set_title('obj_error')
        
        axs[2].plot(np.log(self.data['validation_loss'].dropna()))
        axs[2].set_title('validation_loss')
        plt.show()
        
    def getClipOp(self, max_abs: float=1.0) -> None:
        """Not used for now"""
        with self.graph.as_default():
            obj_reshaped = tf.reshape(self.tf_obj, [2, -1])
            obj_clipped = tf.clip_by_norm(obj_reshaped, max_abs, axes=[0])
            obj_clipped_reshaped = tf.reshape(obj_clipped, [-1])
            clipped = tf.assign(self.tf_obj, obj_clipped_reshaped, name='clip_op')
        return clipped



class tfNearFieldPtychoReconsFromSimulation(tfNearFieldPtychoRecons):
    
    def __init__(self,
                 simulation: NFPtychoSimulation,
                 obj_guess: Optional[np.ndarray] = None,
                 probe_guess: Optional[np.ndarray] = None,
                 probe_recons: bool = False,
                 batch_size: int = 0,
                 n_validation_diffs: int = 0) -> None:
        
        """
        """
        self.simulation = simulation
        self.probe_params = simulation.probe_params
        self.obj_params = simulation.obj_params
        self.det_params = simulation.det_params
        
        self.positions = simulation.positions
        self.diffraction_mods = simulation.diffraction_mods
        self.diffraction_mods_shifted = np.fft.fftshift(self.diffraction_mods, axes=(-1,-2))
        self.prop_kernel = simulation.prop_kernel
        
        self.obj_true = simulation.obj_true
        self.probe_true = simulation.probe_true
        
        self.probe_recons = probe_recons
        if not probe_recons:
            self.probe_guess = self.probe_true
        
        self.setObjProbeGuess(obj_guess, probe_guess)
        
        self.setTrainingAndValidation(n_validation_diffs, batch_size)
        
        # Tensorflow setup
        self.createGraphAndVars()
        self.initDataSet()
        
        self.optimizers_defined = False
        
        self.data = DataFrame(columns=['loss','epoch','obj_error','probe_error','validation_loss','patience'],
                              dtype='float32')
        

