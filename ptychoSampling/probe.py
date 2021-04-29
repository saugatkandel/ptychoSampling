#Author - Saugat Kandel
# coding: utf-8


import numpy as np
from ptychoSampling.utils import utils
from ptychoSampling.wavefront import Wavefront
from typing import Optional, Tuple, Any
import abc
from ptychoSampling.logger import logger

__all__ = ["Probe",
           "Probe3D",
           "CustomProbeFromArray",
           "CustomProbe3DFromArray",
           "EllipticalProbe",
           "FocusCircularProbe",
           "GaussianProbe",
           "GaussianSpeckledProbe",
           "RectangleProbe"]

class Probe(abc.ABC):
    """Abstract class that can be inherited as necessary for arbitrary probe structures.

    Assumes a square probe array.

    Parameters
    ----------
    wavelength : float
        Wavelength of the probe wavefront.
    pixel_size : tuple(float,...)
        Pixel pitch at the sample plane [y, x] (in m).
    shape: tuple(int,...)
        Number of pixels in the [y, x] sides of the probe array.
    n_photons : float
        Total number of photons in the probe wavefront, at the sample plane.
    defocus_dist : float, optional
        Distance to further propagate the probe once the initial structure is defined (in m). For example,
        if we want to simulate the probe beam due to a square aperture close to the sample, then we can first create
        the square probe structure (exit wave from the aperture), and then use this `defocus_dist` parameter to
        propagate the wavefront to the sample plane. Default is set to 0 m.
    center_dist: tuple(float, float), optional
        Displacement (along the y-axis and x-axis respectively) of the center of the probe wavefront from the center of
        the pixellation in the sample plane (in m). Defaults to :math:`(0,0)`.
    center_npix: tuple(int, int), optional
        Displacement (along the y-axis and x-axis respectively) of the center of the probe wavefront from the center of
        the pixellation in the sample plane (in pixels). Converted to `center_dist` when necessary. Defaults to
        :math:`(0,0)`.
    width_dist: tuple(float, float), optional
        Width of the probe along the y and x-axes (in m). This is used in the inherited subclasses. Default value is
        :math:`(0,0)`.
    width_npix: tuple(int, int), optional
        Width of the probe in pixels. This is converted to `width_dist` when necessary.
    check_propagation_with_gaussian_fit : bool, optional
        **Experimental**: enables a very crude sanity check to ensure that the defocus required is within the supported
        regimes. Fits a gaussian to the probe function to calculate the *feature size*, which is then used to check
        which propagation method is appropriate.
    apodize: bool, optional
        Apply a Hanning window to the probe structure so that the probe edges go to zero. The probe is re-scaled
        after applying the window function to ensure that the number of photons is the provided number.

    Attributes
    ----------
    wavelength, pixel_size, n_photons, defocus_dist, center_dist, center_npix, width_dist, width_npix : see Parameters
    photons_flux : float
        Average number of photons per pixel (at the sample plane).
    wavefront : ndarray(complex)
        Probe wavefront at the sample plane, after any supplied defocus.
    check_propagation_with_gaussian_fit : bool, optional

    """
    def __init__(self,
                 wavelength: float,
                 pixel_size: Tuple[float, ...],
                 shape: Tuple[int, ...],
                 n_photons: float,
                 defocus_dist: float = 0,
                 center_dist: Tuple[float,float] = (0,0),
                 width_dist: Tuple[float, float] = (0,0),
                 center_npix: Tuple[int,int] = None,
                 width_npix: Tuple[int, int] = None,
                 check_propagation_with_gaussian_fit: bool = False,
                 apodize: bool = False) -> None:
        self.wavelength = wavelength
        self.shape = shape
        self.pixel_size = pixel_size
        self.n_photons = n_photons
        self.defocus_dist = defocus_dist
        self.center_dist = center_dist
        self.width_dist = width_dist
        self.apodize = apodize

        if center_npix is not None:
            logger.warning('If center_npix is supplied, then any supplied center_dist is ignored.')
            self.center_npix = center_npix
            self.center_dist = np.array(center_npix) * np.array(self.pixel_size)
        if width_npix is not None:
            logger.warning('If width_npix is supplied, then any supplied width_dist is ignored.')
            self.width_npix = width_npix
            self.width_dist = np.array(width_npix) * np.array(self.pixel_size)

        #wavefront_array = np.zeros((npix, npix), dtype='complex64')

        #self.wavefront = propagators.Wavefront(wavefront_array,
        self.wavefront = Wavefront(np.zeros(shape),
                                   wavelength=wavelength,
                                   pixel_size=pixel_size)

        self.photons_flux = n_photons / (shape[-1] * shape[-2])
        self.check_propagation_with_gaussian_fit = check_propagation_with_gaussian_fit

    @abc.abstractmethod
    def _calculateWavefront(self) -> None:
        """Abstract method that, when inherited, should calculate the probe wavefront from the supplied parameters."""
        pass

    @property
    def gaussian_fit(self) -> dict:
        r"""Fit a 2-d gaussian to the probe intensities (not amplitudes) and return the fit parameters.

        The returned dictionary contains the following fit parameters (as described in [1]_):
            * ``amplitude`` : Amplitude of the fitted gaussian.
            * ``center_x`` : X-offset of the center of the fitted gaussian.
            * ``center_y`` : Y-offset of the center of the fitted gaussian.
            * | ``theta`` : Clockwise rotation angle for the gaussian fit. Prior to the rotation, primary axes of the \
              | gaussian  are aligned to the X and Y axes.
            * ``sigma_x`` : Spread (standard deviation) of the gaussian fit along the x-axis (prior to rotation).
            * ``sigma_y`` : Spread of the gaussian fit along the y-axis (prior to rotation).
            * | ``offset`` : Constant level of offset applied to the intensity throughout the probe array. This could,
              | for instance, represent the level of background radiation.

        Returns
        -------
        out : dict
            Dictionary containing the fit parameters.

        See also
        --------
        _calculateGaussianFit

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
        """
        if not hasattr(self, '_gaussian_fit'):
            self._calculateGaussianFit()
        return self._gaussian_fit

    @property
    def gaussian_fwhm(self) -> Tuple[float, float]:
        r"""Fit a 2-d gaussian the the probe intensities (not amplitudes) and return the FWHM along the primary axes.

        The full width at half-maximum (FWHM) is calculated as :math:`\text{FWHM}_x = \sigma_x * 2.355` and similarly for
        :math:`\textrm{FWHM}_y`.

        Returns
        -------
        out : Tuple[float, float]
            FWHM along the primary axes.

        See also
        --------
        gaussian_fit
        _calculateGaussianFit
        """
        if not hasattr(self, '_gaussian_fwhm'):
            self._calculateGaussianFit()
        return self._gaussian_fwhm


    def _calculateGaussianFit(self) -> None:
        r"""Fit a 2d gaussian to the probe intensities.

        Performs a least-squares fit (using ``scipy.optimize.curve_fit``) to fit a 2d gaussian to the probe
        intensities. Uses the calculated gaussian spread to calculate the FWHM as well.

        See also
        --------
        gaussian_fit
        utils.generalized2dGaussian
        """
        logger.info('Fitting a generalized 2d gaussian to the probe intensity.')
        from scipy.optimize import curve_fit

        nx = self.shape[-1]
        ny = self.shape[-2]
        #intensities = np.fft.ifftshift(np.abs(self.wavefront)**2)
        intensities = self.wavefront.fftshift.intensities
        y = np.arange(-ny // 2, ny // 2) * self.pixel_size[0]
        x = np.arange(-ny // 2, nx // 2) * self.pixel_size[1]
        yy, xx = np.meshgrid(y, x)
        xdata = np.stack((xx.flatten(), yy.flatten()), axis=1)
        bounds_min = [0, x[0], y[0], 0, 0, -np.pi/4, 0]
        bounds_max = [intensities.sum(), x[-1], y[-1], x[-1] * 2, y[-1] * 2, np.pi/4, intensities.max()]
        popt, _ = curve_fit(utils.generalized2dGaussian,
                            xdata,
                            intensities.flatten(),
                            bounds=[bounds_min, bounds_max])
        amplitude, center_x, center_y, sigma_x, sigma_y, theta, offset = popt
        self._gaussian_fit = {"amplitude": amplitude,
                             "center_x": center_x,
                             "center_y": center_y,
                             "sigma_x": sigma_x,
                             "sigma_y": sigma_y,
                             "theta": theta,
                             "offset": offset}
        self._gaussian_fwhm = 2.355 * np.array((sigma_y, sigma_x))


    def _propagateWavefront(self) -> None:
        """Propagate the probe wavefront by `defocus_dist`, and, if the gaussian fit has been priorly calculated,
        recalculate the fit.
        Returns
        -------
        None
        """

        if self.defocus_dist > 0:
            if self.check_propagation_with_gaussian_fit:
                self._checkPropGaussianFit()
            else:
                logger.debug("Experimental feature: The propagator assumes that the defocus distance is within the "
                             + "regime of Fresnel propagation. The user must ensure beforehand that this is True. "
                            + "A crude version of this check can be achieved by setting the parameter "
                            + "'check_propagation_with_gaussian_fit' to True when instantiating the Probe class.")

            #self.wavefront = propagators.propTF(wavefront=self.wavefront,
            #                                    prop_dist=self.defocus_dist)
            self.wavefront = self.wavefront.propTF(prop_dist=self.defocus_dist)
            if self.apodize:
                self.wavefront = self._apodizeWavefront(self.wavefront, self.n_photons)
            if hasattr(self, '_gaussian_fit'):
                self._calculateGaussianFit()

    @staticmethod
    def _apodizeWavefront(wavefront: Wavefront, n_photons) -> None:
        """Apodization using a Hanning window followed by a rescaling."""
        windowy = np.hanning(wavefront.shape[0])
        windowx = np.hanning(wavefront.shape[1])
        window_2d = windowy[:,None] * windowx[None, :]
        wv_new = (wavefront.fftshift * window_2d).fftshift
        wv_scaled = wv_new * np.sqrt(n_photons / wv_new.intensities.sum())
        return wv_scaled

    def _checkPropGaussianFit(self) -> bool:
        """Experimental feature. To be used primarily for any debugging."""
        from ptychoSampling.wavefront import  checkPropagationType

        logger.warn("the check for propagation type is an experimental feature.")
        self._calculateGaussianFit()
        feature_size = np.array(self._gaussian_fwhm) * 2

        propagation_type = checkPropagationType(wavelength=self.wavelength,
                                                prop_dist=self.defocus_dist,
                                                source_pixel_size=self.pixel_size,
                                                max_feature_size=feature_size)
        errors = {-1: "Propagation type is different along the x and y directions. This is not supported.",
                  0: "Defocus distance is too small (Fresnel number too high) for transfer function method.",
                  2: "Defocus distance too large. Only near field defocus supported."}
        if propagation_type in errors:
            e = ValueError(errors[propagation_type])
            logger.error(e)
            raise e
        return True

class CustomProbeFromArray(Probe):
    r"""Create a Probe object using a supplied wavefront.

    See documentation for `Probe` for information on the attributes.

    Parameters
    ----------
    wavefront_array : array_like(complex)
        Square 2D array that contains the probe wavefront.
    wavelength, pixel_size, defocus_dist : see documentation for `Probe`.

    See also
    --------
    Probe
    """
    def __init__(self, wavefront_array: np.ndarray,
                 wavelength: float,
                 pixel_size: Tuple[float, float],
                 defocus_dist: float=0) -> None:
        super().__init__(wavelength=wavelength,
                         pixel_size=pixel_size,
                         shape=wavefront_array.shape,
                         n_photons=np.sum(np.abs(wavefront_array)**2),
                         defocus_dist=defocus_dist)
        #self.wavefront = self.wavefront.update(array = wavefront_array.copy())
        self.wavefront = Wavefront(wavefront_array,
                                   wavelength=self.wavefront.wavelength,
                                   pixel_size=self.wavefront.pixel_size)
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:
        """Simply propagates the supplied wavefront by `defocus_dist`."""
        self._propagateWavefront()


class RectangleProbe(Probe):
    r"""Create a Probe object using the exit wave from a rectangular aperture.

    See documentation for `Probe` information on other parameters and attributes.

    Parameters
    ----------
    *args : Any
        Parameters required to initialize a `Probe` object. Supplied to the parent `Probe` class.
    width_dist : tuple(float, float)
        Width of the rectangular aperture in the x and y directions respectively (in m).
    width_npix : tuple(int, int)
        Width of rectangular aperture in x and y directions (in pixel values).
    **kwargs : Any
        Optional arguments for a `Probe` object. Supplied to the parent `Probe` class.
    Attributes
    ----------
    width_dist, width_npix : see Parameters

    See also
    --------
    Probe
    """
    def __init__(self, *args: Any,
                **kwargs: Any):
        super().__init__(*args,
                         **kwargs)
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:
        """Calculates and propagates the exit wave from a rectangular aperture of the supplied dimensions."""
        nx = self.shape[-1]
        ny = self.shape[-2]

        x = np.arange(-nx // 2, nx // 2) * self.pixel_size[1]
        y = np.arange(-ny // 2, ny // 2)[:, np.newaxis] * self.pixel_size[0]
        xarr = np.where(np.abs(x - self.center_dist[1]) <= self.width_dist[1] / 2, 1, 0)
        yarr = np.where(np.abs(y - self.center_dist[0]) <= self.width_dist[0] / 2, 1, 0)
        array = np.fft.fftshift((xarr * yarr).astype('complex64'))

        scaling_factor = np.sqrt(self.n_photons / (np.abs(array)**2).sum())
        #self.wavefront = self.wavefront.update(array = scaling_factor * array)
        self.wavefront = Wavefront(scaling_factor * array, wavelength=self.wavelength, pixel_size=self.pixel_size)
        self._propagateWavefront()

class EllipticalProbe(Probe):
    r"""Create a Probe object using the exit wave from a elliptical aperture.

        Also see documentation for `Probe` information on other parameters and attributes.

        Parameters
        ----------
        width_dist : tuple(float, float)
            Diameter (?) of the aperture in y and x directions respectively(in m).
        width_npix : tuple(int, int)
            Diameter (?) of the aperture in y and x directions respectively (in pixels).
        *args : Any
            Parameters required to initialize a `Probe` object. Supplied to the parent `Probe` class.
        **kwargs : Any
            Optional arguments for a `Probe` object. Supplied to the parent `Probe` class.

        Attributes
        ----------
        width_dist, width_npix : see Parameters

        See also
        --------
        Probe

        Notes
        -----
        If :math:`(2b, 2a)` is the `width_dist` parameter, then an ellipse is defined as:

        .. math:: \frac{x^2}{a^2} +

        the :math:`\text{circ}` function for the aperture. If :math:`r` is the radius of the aperture and
        :math:`a` is the distance of a point :math:`(x,y)` from the center of the aperture, this is defined as:

        .. math:: \frac{x^2}{a^2} + \frac{y^2}{b^2} = 1

        Then a point :math:`(x_1,y_1)` can be said to be inside the ellipse when:

        .. math:: x_1^2 b^2 + y_1^2 a^2 <= a^2 b^2

        For a circular probe of radius :math:`r`, we have :math:`a=b=r`, such that:

        .. math:: x_1^2 + y_1^2 <= r^2

        """
    def __init__(self, *args: Any,
                 **kwargs: Any):
        super().__init__(*args,
                         **kwargs)
        self._calculateWavefront()

    def _calculateWavefront(self):
        """Calculates and propagates the exit wave from a circular aperture of the supplied radius.
        """
        a = self.width_dist[1] / 2
        b = self.width_dist[0] / 2
        nx = self.shape[1]
        ny = self.shape[0]
        x = np.arange(-nx// 2, nx// 2) * self.pixel_size[1]
        y = np.arange(-ny// 2, ny// 2)[:, np.newaxis] * self.pixel_size[0]

        lhs = b**2 * (x - self.center_dist[1])**2 + a**2 * (y - self.center_dist[0])**2
        rhs = a**2 * b**2

        wavefront_array = np.zeros(lhs.shape, dtype='complex64')
        wavefront_array[lhs <= rhs] = 1.0
        wavefront_array = np.fft.fftshift(wavefront_array)
        scaling_factor = np.sqrt(self.n_photons / (np.abs(wavefront_array)**2).sum())
        #self.wavefront = self.wavefront.update(array=scaling_factor * wavefront_array)
        self.wavefront = Wavefront(scaling_factor * wavefront_array,
                                   wavelength=self.wavelength,
                                   pixel_size=self.pixel_size)
        self._propagateWavefront()
        

class GaussianProbe(Probe):
    r"""Create a Probe object with a gaussian wavefront.

    Also see documentation for `Probe` information on other parameters and attributes.

    Parameters
    ----------
    width_dist : tuple(float, float)
        Spread (standard deviation) of the gaussian in the y and x directions respectively(in m). Typically denoted
        by *sigma* (:math:`\sigma`).
    width_npix : tuple(int, int)
        Same as `width_dist`, but in pixels.
    theta : float, optional
        Angle with which to clockwise rotate the primary axes of the 2d gaussian after generation.
    *args : Any
        Parameters required to initialize a `Probe` object. Supplied to the parent `Probe` class.
    **kwargs : Any
        Optional arguments for a `Probe` object. Supplied to the parent `Probe` class.

    Attributes
    ----------
    width_dist, width_npix, theta : see Parameters

    See also
    --------
    Probe
    utils.generalized2dGaussian

    Notes
    -----
    The probe wavefront is generated with constant phase and with intensity calculated according to the equation [2]_:

        .. math:: f(x,y) = A \exp\left(-a(x-x_0)^2 - 2b(x-x_0)(y-y_0) - c(y-y_0)^2\right),

    where :math:`A` is the amplitude, :math:`a=\cos^2\theta/(2\sigma_x^2) + \sin^2\theta/(2\sigma_y^2)`, with
    :math:`b=-\sin 2\theta/(4\sigma_x^2) + \sin 2\theta/(4\sigma_y^2)`, and with
    :math:`c=\sin^2\theta/(2\sigma_x^2) + \cos^2\theta/(2\sigma_y^2)`.

    References
    ----------
    .. [2] https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

    """
    def __init__(self, *args: Any,
                 theta: float = 0,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.theta = theta
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:
        """Calculate the probe wavefront with constant phase and with a gaussian intensity. Propagate the wavefront."""
        sigma_x = self.width_dist[1]
        sigma_y = self.width_dist[0]

        nx = self.shape[-1]
        ny = self.shape[-2]
        x = np.arange(-nx // 2, nx // 2) * self.pixel_size[1]
        y = np.arange(-ny // 2, ny // 2) * self.pixel_size[0]
        xx, yy = np.meshgrid(x, y)
        xdata = np.stack((xx.flatten(), yy.flatten()), axis=1)
        intensity = utils.generalized2dGaussian(xdata,
                                                amplitude=1,
                                                center_x=self.center_dist[1],
                                                center_y=self.center_dist[0],
                                                sigma_x = sigma_x,
                                                sigma_y = sigma_y,
                                                theta=self.theta,
                                                offset=0).reshape((ny, nx))

        scaling_factor = np.sqrt(self.n_photons / intensity.sum())
        wavefront_array = scaling_factor * np.sqrt(intensity).astype('complex64')
        wavefront_array = np.fft.fftshift(np.reshape(wavefront_array, (ny, nx)))
        #self.wavefront = self.wavefront.update(array=wavefront_array)
        self.wavefront = Wavefront(wavefront_array, wavelength=self.wavelength, pixel_size=self.pixel_size)
        self._propagateWavefront()

class GaussianSpeckledProbe(Probe):
    """Create a gaussian probe, then modulate it with a speckle pattern.

    First calculates a gaussian wavefront, then multiplies it with a speckle pattern for the modulation. The speckle
    pattern is randomly generated and varies between instances of `GaussianSpeckledProbe`.

    Also see documentation for `Probe` information on other parameters and attributes.

    Parameters
    ----------
    width_dist, width_npix, theta : see the documentation for the corresponding parameters in GaussianProbe.
    speckle_window_npix : int
        Aperture size used to generate a speckled pattern (in pixels).
    *args : Any
        Parameters required to initialize a `Probe` object. Supplied to the parent `Probe` class.
    **kwargs : Any
        Optional arguments for a `Probe` object. Supplied to the parent `Probe` class.

    Attributes
    ----------
    width_x, width_y, theta, speckle_window_npix : see Parameters

    Notes
    -----
    In ptychography, probe structures with *phase diversity* (i.e. with large angular ranges) are known to reduce
    the dynamic range of the detected signal by spreading the zero-order light over an extended area of the detector
    [3]_. This makes the reconstruction more robust and increases the rate of convergence of the reconstruction
    algorithm. This is generally accomplished experimentally via a diffuser, which introduces speckles into
    probe wavefront. The *speckled probe* used here thus emulates a gaussian probe wavefront modulated by a diffuser.

    Additionally, for successful near-field ptychography [4]_, it is important to ensure that the diffraction patterns
    generated in consecutive scan positions are sufficiently diverse (or different). This can be accomplished by
    using a probe structure that varies rapidly as we traverse the spatial structure. A speckle pattern accomplishes
    this and can be effectively used for near-field ptychography.

    See also
    --------
    GaussianProbe
    Probe
    utils.getSpeckle

    References
    ----------
    .. [3] Morrison, G. R., Zhang, F., Gianoncelli, A. & Robinson, I. K. X-ray ptychography using randomized zone
        plates. Opt. Express 26, 14915 (2018).
    .. [4] Richard M. Clare, Marco Stockmar, Martin Dierolf, Irene Zanette, and Franz Pfeiffer,
        "Characterization of near-field ptychography," Opt. Express 23, 19728-19742 (2015)
    """
    def __init__(self, *args: Any,
                 speckle_window_npix: int,
                 theta: float = 0,
                 **kwargs: Any) -> None:
        super().__init__(*args,
                         **kwargs)
        self.speckle_window_npix = speckle_window_npix
        self.theta = theta
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:
        """Calculate a wavefront with gaussian intensity, then modulate the wavefront using a speckle pattern."""
        sigma_x = self.width_dist[1]
        sigma_y = self.width_dist[0]

        nx = self.shape[-1]
        ny = self.shape[-2]
        x = np.arange(-nx // 2, nx // 2) * self.pixel_size[1]
        y = np.arange(-ny // 2, ny // 2) * self.pixel_size[0]
        xx, yy = np.meshgrid(x, y)
        intensity = utils.generalized2dGaussian(np.stack((xx.flatten(), yy.flatten()), axis=1),
                                                amplitude=1,
                                                center_x=self.center_dist[0],
                                                center_y=self.center_dist[1],
                                                sigma_x=sigma_x,
                                                sigma_y=sigma_y,
                                                theta=self.theta,
                                                offset=0)
        amplitude = np.sqrt(intensity).astype('complex64').reshape(ny, nx)

        speckle = utils.getSpeckle((ny, nx), self.speckle_window_npix)
        wavefront_array = np.fft.fftshift(amplitude * speckle)
        scaling_factor = np.sqrt(self.n_photons / np.sum(np.abs(wavefront_array)**2))
        #self.wavefront = self.wavefront.update(array=scaling_factor * wavefront_array)
        self.wavefront = Wavefront(scaling_factor * wavefront_array, wavelength=self.wavelength,
                                   pixel_size=self.pixel_size)
        self._propagateWavefront()

class FocusCircularProbe(Probe):
    r"""Create a Probe object containing an airy pattern obtained due to a lens and a circular aperture.

    Given a lens of focal length :math:`f`, if we place a circular aperture at a distance :math:`f` *before*
    the lens, then we obtain an airy pattern at the focus *after* the lens [5]_. The `FocusCircularProbe` class
    simulates such an experimental condition and populates the probe wavefront with the airy pattern obtained. It
    then allows for subsequent defocus of the wavefront.

    The probe wavefront can be generated by either supplying *both* `focal_length` and `aperture_radius`,
    or by supplying `focus_radius_npix`. If `focus_radius_npix` is supplied, then any supplied `focal_length` and
    `aperture_radius` are ignored; the `focal_length` is set to a default value of 10 m, and the aperture radius is
    calculated such that the generated airy pattern in the wavefront has its first minimum at `focus_radius_npix`
    pixels from the center of the pattern.

    If the size of the probe array is not large enough, then the probe generated contains significant aliasing. To
    reduce this aliasing, we can artificially oversample the probe wavefront by increasing the size of the
    probe array. Using this oversampled array to calculate the probe wavefront, we reduce the aliasing. We can then
    the central :math:`\text{npix}\times\text{npix}` pixels of the oversampled array to get the desired wavefront.

    See the `Probe` documentation for information on the other parameters and attributes.

    Notes
    -----
    To describe the steps used for the wavefront calculation, we denote the focal length as :math:`f`, the aperture
    radius as :math:`r_a`, the number of pixels in the probe array as :math:`N`, the pixel pitch at the probe focus
    as :math:`\Delta p_f`, the pixel pitch at the aperture as :math:`\Delta p_a`.

    #. If `focus_radius_npix` (:math:`N_f`) is supplied, then, assume that the resolution (:math:`\delta_r`) at the
        sample plane is equal to the pixel pitch :math:`\delta_r = \Delta p_f`, and that :math:`f=10` m, use the Rayleigh
        resolution criterion to calculate the aperture radius (Equations 4.161-4.163 in [6]_):

            .. math:: r_a = \frac{1.22 \lambda \pi f}{2 \pi N_f \delta_r}

        Otherwise, if the `aperture_radius` is supplied, use the supplied value.

    #. Calculate the pixel pitch at the aperture plane (which is at a distance :math:`f` before the lens):

            .. math:: \Delta p_a = \frac{\lambda f}{N \Delta p_f}

    #. Use the :math:`\text{circ}` function with the aperture radius :math:`r_a` as the wavefront at the aperture
        plane. See documentation for `CircularProbe` for more on the `\text{circ}` function.

    #. Use a fourier transform of this wavefront to get the airy wavefront at the focus plane.

    #. If necessary, propagate by `defocus_dist` to get the wavefront at the sample plane.

    Parameters
    ----------
    focal_length : float, optional
        Focal length of the lens used to generate the probe (in m).
    aperture_radius : float, optional
        Radius of the circular aperture placed *before* the lens, at a distance `focal_length` from the lens.
    focus_radius_npix : int, optional
        Distance (in pixels) of the first minimum of the airy pattern from the center of the pattern.
    oversampling : bool
        Whether to use oversampling to reduce aliasing in the probe wavefront. Default value is `True`.
    oversampling_npix : int
        Number of pixels in each side of the oversampled array. Default value is 1024. If
        :math:`\text{npix} > \text{oversampling_npix}`, then `oversampling_npix` is set to `npix`.
    *args : Any
        Parameters required to initialize a `Probe` object. Supplied to the parent `Probe` class.
    **kwargs : Any
        Optional arguments for a `Probe` object. Supplied to the parent `Probe` class.

    Attributes
    ----------
    focal_length, aperture_radius, focus_radius_npix, oversampling, oversampling_npix : see Parameters

    See also
    --------
    Probe

    References
    ----------
    .. [5] Schmidt, J. D. & Jason D.Schmidt. Numerical Simulation of optical wave propagation (pp. 55-64). (1975).
        doi:10.1117/3.866274
    .. [6] Jacobsen, C. & Kirz, J. X-ray Microscopy_2018_01_15. (2017).

    .. todo::

        Allow for combinations of `focus_radius_npix` with either `focal_length` or `aperture_radius`.

    """
    def __init__(self, *args: Any,
                 focal_length: Optional[float] = None,
                 aperture_radius: Optional[float] = None,
                 oversampling: bool = True,
                 oversampling_npix: int = 1024,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if ((self.width_dist[0] != self.width_dist[1])
                or (self.pixel_size[0] != self.pixel_size[1])
                or (self.shape[0]!= self.shape[1])):
            e = ValueError("Only a circularly symmetric focussed probe is supported. Require equal x and y widths.")
            logger.error(e)
            raise e

        if self.center_dist[0] > 0:
            e = ValueError("Only a centrally located focussed probe is supported (i.e. center at origin). If "
                           + "necessary, we can simply customize the wavefront manually (pad and roll).")
            logger.error(e)
            raise e

        self.focal_length = focal_length
        self.aperture_radius = aperture_radius
        self.oversampling = oversampling
        self.oversampling_npix = oversampling_npix
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:
        """Calculating the airy pattern then propagating it by `defocus_dist`."""
        if self.width_dist[0] > 0:
            logger.warning("Warning: if width at the focus is supplied, "
                           + "any supplied focal_length and aperture radius are ignored.")
            self.aperture_radius = None
            self.focal_length = None
            # Assuming resolution equal to pixel pitch and focal length of 10.0 m.
            focal_length = 10.0
            focus_radius = self.width_dist[0] / 2
            # note that jinc(1,1) = 1.22 * np.pi
            aperture_radius = (self.wavelength * 1.22 * np.pi * focal_length
                               / 2 / np.pi / focus_radius)
        else:
            if None in [self.aperture_radius, self.focal_length]:
                e =  ValueError('Either focus_radius_npix or BOTH aperture_radius and focal_length must be supplied.')
                logger.error(e)
                raise e
            aperture_radius = self.aperture_radius
            focal_length = self.focal_length

        npix_oversampled = max(self.oversampling_npix, self.shape[-1]) if self.oversampling else self.shape[0]
        pixel_pitch_aperture = self.wavelength * focal_length / (npix_oversampled * self.pixel_size[0])

        x = np.arange(-npix_oversampled // 2, npix_oversampled // 2) * pixel_pitch_aperture

        r = np.sqrt(x**2 + x[:,np.newaxis]**2).astype('float32')
        circ_wavefront = np.zeros(r.shape)
        circ_wavefront[r < aperture_radius] = 1.0
        circ_wavefront[r == aperture_radius] = 0.5

        probe_vals = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ_wavefront), norm='ortho'))

        n1 = npix_oversampled // 2 - self.shape[0] // 2
        n2 = npix_oversampled // 2 + self.shape[1] // 2

        scaling_factor = np.sqrt(self.n_photons / np.sum(np.abs(probe_vals)**2))
        wavefront_array = probe_vals[n1:n2, n1:n2].astype('complex64') * scaling_factor
        wavefront_array = np.fft.fftshift(wavefront_array)
        #self.wavefront = self.wavefront.update(array=wavefront_array)
        self.wavefront = Wavefront(wavefront_array, wavelength=self.wavelength, pixel_size=self.pixel_size)
        self._propagateWavefront()

class Probe3D(Probe, abc.ABC):
    """3D probe structure. For use in bragg ptychography experiments (etc).

    .. todo::

        Addressing 3D probe structures adequately is a lot harder than addressing 2D probe structures. For example,
        we need to address:

            * rotation and interpolation to generate probe structures (in the lab reference frame) at various angles of
                incidence.
            * the pixel sizes along the x, y, and z dimensions (in various reference frames). Even when
                :math:`2\theta = 90^\circ`, I am not sure how to properly calculate the pixel size along the probe
                propagation direction.

        As a consequence, the approach here is very ad-hoc.
    """
    def __init__(self, wavelength: float,
                 shape: Tuple[int, int, int],
                 pixel_size: Tuple[float, float, float] = None,
                 photons_flux: float = None):

        super().__init__(wavelength=wavelength, shape=shape, pixel_size=pixel_size, n_photons=np.inf)
        self.photons_flux = photons_flux


class CustomProbe3DFromArray(Probe3D):
    def __init__(self,
                 array: np.ndarray,
                 wavelength: float,
                 pixel_size: Tuple[float, float, float] = None,
                 photons_flux: float=None):
        super().__init__(wavelength, array.shape, pixel_size, photons_flux)
        self.wavefront = Wavefront(array, wavelength=wavelength)

    def _calculateWavefront(self):
        pass


