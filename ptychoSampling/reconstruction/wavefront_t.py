import tensorflow as tf
import numpy as np
from typing import Union, Tuple
from ptychoSampling.logger import  logger

def ununsed_fftshift_t(t: tf.Tensor,
               axes: Union[int, Tuple[int,...]] = (-1,-2),
               name: str = None):
    """ Does not work on the gpu!!

    Shift the zero-frequency component to the center of the spectrum. (Adapted from tensflow 2.0 source code)

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.
    @compatibility(numpy)
    Equivalent to numpy.fft.fftshift.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftshift.html
    @end_compatibility
    For example:
    ```python
    x = tf.signal.fftshift([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
    x.numpy() # array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    ```
    Parameters
    -----------
    t : tensor
        Input tensor.
    axes : int or shape `tuple`
        Optional Axes over which to shift.  Default is (-1,-2), which shifts the last two axes.
    name : str
        An optional name for the operation.
    Returns
    --------
    out : tensor
        The shifted tensor.
    """
    with tf.name_scope(name, "fftshift") as name:
        if isinstance(axes, int):
            shift = int(t._shape_tuple()[axes] // 2)
        else:
            shift = [int((t._shape_tuple()[ax]) // 2) for ax in axes]

        return tf.roll(t, shift, axes)

def batch_fftshift2d(tensor: tf.Tensor):
    """Fftshift using split op. Only works for arrays with even number of elements in the last two axes."""
    # Shifts high frequency elements into the center of the filter
    indexes = len(tensor.get_shape()) - 1
    top, bottom = tf.split(tensor, 2, axis=indexes)
    tensor = tf.concat([bottom, top], indexes )
    left, right = tf.split(tensor, 2, axis=indexes - 1)
    tensor = tf.concat([right, left], indexes - 1 )
    return tensor

def fftshift_t(tensor: tf.Tensor) -> tf.Tensor:
    return batch_fftshift2d(tensor)

def ifftshift_t(tensor: tf.Tensor) -> tf.Tensor:
    return batch_fftshift2d(tensor)

def unused_ifftshift_t(t: tf.Tensor, axes=(-1,-2), name=None):
    """The inverse of fftshift.

    Although identical for even-length x,
    the functions differ by one sample for odd-length x.
    @compatibility(numpy)
    Equivalent to numpy.fft.ifftshift.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.ifftshift.html
    @end_compatibility
    For example:
    ```python
    x = tf.signal.ifftshift([[ 0.,  1.,  2.],[ 3.,  4., -4.],[-3., -2., -1.]])
    x.numpy() # array([[ 4., -4.,  3.],[-2., -1., -3.],[ 1.,  2.,  0.]])
    ```
    Parameters
    ----------
    t : tensor
        input tensor.
    axes : `int` or shape `tuple`
        Axes over which to calculate. Defaults to (-1, -2), which shifts the last two axes.
    name : str
        An optional name for the operation.
    Returns
    -------
    out : tensor
        The shifted tensor.
    """
    with tf.python.ops.name_scope(name, "fftshift") as name:
        if isinstance(axes, int):
            shift = -int(t._shape_tuple()[axes] // 2)
        else:
            shift = [-int((t._shape_tuple()[ax]) // 2) for ax in axes]
        return tf.roll(t, shift, axes)

def fft2_t(t: tf.Tensor):
    shape = t.get_shape().as_list()
    norm = (shape[-1] * shape[-2]) ** 0.5
    return tf.fft2d(t) / norm

def ifft2_t(t: tf.Tensor):
    shape = t.get_shape().as_list()
    norm = (shape[-1] * shape[-2]) ** 0.5
    return tf.ifft2d(t) * norm

def propFF_t(t: tf.Tensor,
             apply_phase_factor: bool = False,
             wavelength: float = None,
             pixel_size: float = None,
             prop_dist: float = None,
             reuse_phase_factor: bool = False,
             quadratic_phase_factor: tf.Tensor = None,
             return_quadratic_phase_factor: bool = False,
             backward=False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Fraunhofer propagation of the supplied wavefront.

    Parameters
    ----------
    apply_phase_factor : bool
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
    ny = t._shape_tuple()[-2]
    nx = t._shape_tuple()[-1]

    if None not in [prop_dist, wavelength, pixel_size]:
        new_pixel_size = (wavelength * prop_dist / (nx * pixel_size[0]),
                          wavelength * prop_dist / (ny * pixel_size[1]))

    if apply_phase_factor:
        if quadratic_phase_factor is None:
            quadratic_phase_factor = tf.zeros_like(t)
        if not reuse_phase_factor:
            k = 2 * np.pi / wavelength

            # reciprocal space pixel size
            rdx = pixel_size[0] if backward else new_pixel_size[0]
            rdy = pixel_size[1] if backward else new_pixel_size[1]

            # reciprocal space coordinates
            x = tf.range(-nx // 2, nx // 2) * rdx
            y = tf.range(-ny // 2, ny // 2)[:, None] * rdy

            # quadratic phase factor
            q = fftshift_t(tf.exp(1j * k / (2 * prop_dist) * (x ** 2 + y ** 2)))
            quadratic_phase_factor = q
    else:
        quadratic_phase_factor = tf.ones_like(t)

    if backward:
        new_t = ifft2_t(t / quadratic_phase_factor)
    else:
        new_t = fft2_t(t) * quadratic_phase_factor

    if return_quadratic_phase_factor:
        return new_t, quadratic_phase_factor
    return new_t

def propTF_t(t: tf.Tensor,
             wavelength: float = None,
             pixel_size: Tuple[float, float] = None,
             prop_dist: float = None,
             reuse_transfer_function: bool = False,
             transfer_function : tf.Tensor = None,
             return_transfer_function: bool = False,
             backward: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Propagation of the supplied wavefront using the Transfer Function function.

    This propagation method is also referred to as *angular spectrum* or *fresnel* propagation.

    Parameters
    ----------
    wavefront : array_like(complex)
        Wavefront to be propagated.
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
    if transfer_function is None:
        transfer_function = tf.zeros_like(t, dtype='complex64')
    if not reuse_transfer_function:
        nx = t._shape_tuple()[-1]
        ny = t._shape_tuple()[-2]

        k = 2 * np.pi / wavelength

        # reciprocal space pixel size
        rdx = wavelength * prop_dist / (nx * pixel_size[0])
        rdy = wavelength * prop_dist / (ny * pixel_size[1])

        # reciprocal space coords
        x = tf.range(-nx // 2, nx // 2, dtype='float32') * rdx
        y = tf.range(-ny // 2, ny // 2, dtype='float32')[:, None] * rdy

        phase = - k / (2 * prop_dist) * (x ** 2 + y ** 2)
        H = fftshift_t(tf.exp(1j * tf.cast(phase, 'complex64')))
        transfer_function = H

    if backward:
        out = fft2_t(ifft2_t(t) / transfer_function)
    else:
        out = ifft2_t(transfer_function * fft2_t(t))
    if return_transfer_function:
        return out, transfer_function
    return out





