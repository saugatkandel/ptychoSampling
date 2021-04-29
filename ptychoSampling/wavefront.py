import numpy as np
from ptychoSampling.logger import logger
from typing import Tuple

__all__ = ["Wavefront", "fft2", "ifft2"]

class Wavefront(np.ndarray):
    """
    Attributes
    ----------
    wavelength, pixel_size : See `Parameters`.
    """
    def __new__(cls, input_array: np.ndarray,
                wavelength: float=None,
                pixel_size: Tuple[float,float]=None):
        """
        Parameters
        ----------
        input_array : array_like(complex)
            Values with which we populate the wavefront.
        wavelength : float
            Wavelength (in m).
        pixel_size : tuple(float, float)
            Pixel size [y, x] (in m).

        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.wavelength = wavelength
        obj.pixel_size = pixel_size
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.wavelength = getattr(obj, 'wavelength', None)
        self.pixel_size = getattr(obj, 'pixel_size', None)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Wavefront, self).__reduce__()
        # Create our own tuple to pass to __setstate__, but append the __dict__ rather than individual members.
        new_state = pickled_state[2] + (self.__dict__,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        self.__dict__.update(state[-1])  # Update the internal dict from state
        # Call the parent's __setstate__ with the other tuple elements.
        super(Wavefront, self).__setstate__(state[0:-1])

    #@staticmethod
    def fft2(self):#wv: 'Wavefront'):
        return fft2(self)

    def ifft2(self):#wv: 'Wavefront'):
        return ifft2(self)
        #out = np.fft.ifft2(self, norm='ortho')
        #return Wavefront(out, wavelength=self.wavelength, pixel_size=self.pixel_size)

    @property
    def phase(self):
        return np.asarray(np.angle(self), dtype='float32')

    @property
    def amplitude(self):
        return np.asarray(np.abs(self), dtype='float32')

    @property
    def intensities(self):
        return self.amplitude**2

    @property
    def fftshift(self):
        return Wavefront(np.fft.fftshift(self, axes=(-1, -2)), pixel_size=self.pixel_size, wavelength=self.wavelength)

    @property
    def ifftshift(self):
        return Wavefront(np.fft.ifftshift(self, axes=(-1, -2)), pixel_size=self.pixel_size, wavelength=self.wavelength)

    def propFF(self, apply_phase_factor: bool = False,
               prop_dist: float = None,
               reuse_phase_factor: bool = False,
               quadratic_phase_factor: np.ndarray = None,
               backward=False) -> 'Wavefront':
        """Fraunhofer propagation of the supplied wavefront.

        Parameters
        ----------
        apply_phase_factor: bool
            Whether to apply the quadratic phase factor. The phase factor is not necessary if we only want the wavefront
            at the detector plane to calculate the intensity patterns recorded. If this is set to `True`, then we need to
            supply either the quadratic phase factor for the `quadratic_phase_factor` parameter, or a `Wavefront`
            object with the `pixel_size` and `wavelength` attributes as well as the `prop_dist` function parameter.
        prop_dist : float
            Propagation distance (in m). Only required when `apply_quadratic_phase_factor` is `True` and
            `quadratic_phase_factor` is `None`.
        reuse_phase_factor : bool
            Whether to reuse the quadratic phase factor supplied through the `quadratic_phase_factor` parameter. Default
            is `False`. If set to `True`, then the quadratic phase factor must be supplied through the
            `quadratic_phase_factor` parameter.
        quadratic_phase_factor : array_like(complex)
            Use the supplied quadratic phase factor. Only required when `apply_quadratic_phase_factor` is `True`. The
            function either reuses or mutates (updates) this array, depending on the 'reuse_quadratic_phase_factor`
            parameter. This parameter can be used when we want to avoid recalculating the phase factor for multiple
            propagations for the same experimental parameters.
        backward : bool
            Propagate backward instead of forward.
        Returns
        -------
        out_wavefront : Wavefront
            Output wavefront.
        """

        new_pixel_size = None
        nx = self.shape[-1]
        ny = self.shape[-2]


        if None not in [prop_dist, self.wavelength, self.pixel_size]:
            new_pixel_size = (self.wavelength * prop_dist / (ny * self.pixel_size[0]),
                              self.wavelength * prop_dist / (nx * self.pixel_size[1]))
            #new_pixel_size = self.wavelength * prop_dist / (npix * self.pixel_size)

        if apply_phase_factor:
            if quadratic_phase_factor is None:
                quadratic_phase_factor = np.zeros((ny, nx), dtype='complex64')
            if not reuse_phase_factor:
                k = 2 * np.pi / self.wavelength

                # reciprocal space pixel size
                rdy, rdx = self.pixel_size if backward else new_pixel_size

                # reciprocal space coordinates
                x = np.arange(-nx // 2, nx // 2) * rdx
                y = np.arange(-ny // 2, ny // 2)[:,np.newaxis] * rdy

                # quadratic phase factor
                q = np.fft.fftshift(np.exp(1j * k / (2 * prop_dist) * (x ** 2 + y ** 2)))
                quadratic_phase_factor[:] = q[:]
        else:
            quadratic_phase_factor = np.ones((ny, nx), dtype='complex64')

        if backward:
            new_wavefront = (self / quadratic_phase_factor).ifft2()
        else:
            new_wavefront = self.fft2() * quadratic_phase_factor

        new_wavefront.pixel_size = new_pixel_size
        return new_wavefront

    def propTF(self,
               prop_dist: float = None,
               reuse_transfer_function: bool = False,
               transfer_function: np.ndarray = None,
               backward: bool = False) -> 'Wavefront':
        """Propagation of the supplied wavefront using the Transfer Function function.

        This propagation method is also referred to as *angular spectrum* or *fresnel* propagation.

        Parameters
        ----------
        reuse_transfer_function : bool
            Reuse provided transfer function.
        transfer_function : array_like(complex)
            Transfer function after fftshift of reciprocal coordinates.
        prop_dist : float
            Propagation distance (in m).
        backward : bool
            Backward propagation.
        Returns
        -------
        wavefront : Wavefront
            Wavefront after propagation
        """

        nx = self.shape[-1]
        ny = self.shape[-2]

        if transfer_function is None:
            transfer_function = np.zeros((ny, nx), dtype='complex64')
        if not reuse_transfer_function:
            k = 2 * np.pi / self.wavelength

            # reciprocal space pixel size
            rdy = self.wavelength * prop_dist / (ny * self.pixel_size[0])
            rdx = self.wavelength * prop_dist / (nx * self.pixel_size[1])

            # reciprocal space coords
            x = np.arange(-nx // 2, nx // 2) * rdx
            y = np.arange(-ny // 2, ny // 2)[:, np.newaxis] * rdy

            H = np.fft.fftshift(np.exp(-1j * k / (2 * prop_dist) * (x ** 2 + y ** 2)))
            transfer_function[:] = H[:]

        if backward:
            out = (self.ifft2() / transfer_function).fft2()
        else:
            out = (transfer_function * self.fft2()).ifft2()
        return out




def fft2(wv: Wavefront):
    out = np.fft.fft2(wv, norm='ortho')
    return Wavefront(out, pixel_size=wv.pixel_size, wavelength=wv.wavelength)

def ifft2(wv: Wavefront):
    out = np.fft.ifft2(wv, norm='ortho')
    return Wavefront(out, pixel_size=wv.pixel_size, wavelength=wv.wavelength)

def checkPropagationType(wavelength: float,
                         prop_dist: float,
                         source_pixel_size: Tuple[float, float],
                         max_feature_size: Tuple[float, float] = None):
    """
    DO NOT USE IN PRODUCTION CODE.
    This is an experimental feature.
    Check the parameters to decide whether to use farfield, or transfer function.

    Calculates the Fresnel number to decide whether to use nearfield or farfield propagation. Then,
    for the near-field, we use the critical sampling criterion to decide whether to use the transfer function (TF)
    method (if :math:`\Delta x > \lambda z/L`) [1]_.

    Notes
    -----
    Following [1]_, we calculate the Fresnel number :math:`F_N = w^2/\lambda z`, where :math:`w` is the half-width of
    the maximum feature size in the source plane (e.g the half-width of a square aperture, or the radius of a
    circular aperture), :math:`\lambda` is the wavelength, and :math:`z` is the propagation distance. If
    :math:`F_N < 0.1`, we can safely use the Fraunhofer (far-field) method, otherwise we need to use the
    near-field method.

    When the maximum feature size is not provided, the function uses a minimal estimate---twice the pixel size at the
    source plane, or two pixels. This assumes that Fraunhofer propagation is the ost likely propagation method.


    Parameters
    ----------
    wavelength : float
        Wavelength of the source wavefront (in m).
    prop_dist : float
        Propagation distance (in m).
    source_pixel_size : tuple(float, float)
        Pixel pitch [y, x] (in m).
    max_feature_size : tuple(float, float) optional
        Maximum feature size [y, x] (e.g. diameter of a circular aperture) (in m) along the two coordinate axes. For
        the default value (`None`), the function assumes that the features are two pixels wide, at maximum.
    Returns
    -------
    out : int
        Type of propagation. Returns `0` if the propagation distance is too small, with Fresnel number :math:`>50`. For
        this case, the transfer function propagator does not apply, and thus the propagation is not supported.
        Returns `1` if the Fresnel number is between :math:`0.1` and :math:`50`, in which case we can use the
        transfer function method. Returns `2` if the Fresnel number is :math:`<0.1`, in which case we can use the
        Fraunhofer propagation method. Returns `-1` if the propagation type is different for different axes.
    References
    ----------
    .. [1] Voelz, D. "Computational Fourier Optics: A MATLAB Tutorial". doi:https://doi.org/10.1117/3.858456

    """
    max_feature_size = 2 * np.array(source_pixel_size) if max_feature_size is None else np.array(max_feature_size)
    feature_half_width = max_feature_size / 2
    fresnel_number = feature_half_width ** 2 / (wavelength * prop_dist)

    if np.all(fresnel_number > 50):
        prop_type = 0
    elif np.all(fresnel_number > 0.1):
        prop_type = 1
    elif np.all(fresnel_number < 0.1):
        prop_type = 2
    else:
        prop_type = -1
    return prop_type