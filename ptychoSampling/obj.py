import numpy as np
from typing import Tuple, List, Union, Any, Optional
import abc
import scipy, scipy.stats
import skimage, skimage.data, skimage.transform
from ptychoSampling.logger import logger
import ptychoSampling.utils

__all__ = ["Obj",
           "CustomObjFromArray",
           "Simulated2DObj",
           "Simulated3DCrystalCell"]

class Obj(abc.ABC):
    """
    Parameters
    ----------
    shape : Tuple(int, int)
        Number of pixels in each axis of the sample
    border_shape : array_like
        Number of pixels to add along the border in the left, right, top, and bottom margins of the sample. If
        ``border_npix = 10``, then the simulation adds 10 pixels to the left of the sample, and 10 pixels to the
        right of the sample, i.e. a total of 20 pixels along the x-direction (and similarly with y). Defaults to ``0``.
        The provided value needs to be compatible with the `pad` argument for `numpy.pad`.
    border_const : complex
        Constant value to fill the border with. Defaults to ``0``.
    mod_range : float
        Maximum value of the modulus for the sample pixels.
    phase_range : float
        Maximum value of the phase for the sample pixels.
    pixel_size : tuple(float)
        Pixel size at the sample plane (assuming square pixels).

    Attributes
    ----------
    shape, border_shape, border_const, mod_range, phase_range : see Parameters
    array : ndarray(complex64)
        Transmission function for the sample/sample. Shape as defined by the `shape` parameter.
    bordered_array : ndarray(complex64)
        Transmission function for the sample plus the specified border.

    Notes
    -----
    Problematically, in blind ptychography experiments we encounter the *affine phase ambiguity*, i.e. a phase
    gradient (or phase ramp) in both the probe and the sample phases. To avoid this ambiguity, we can add a *known*
    border to the sample (a tight support), and fill the border with ones (i.e,. assume the border is transparent).
    This is the *bright-field boundary condition* that Albert Fannjiang talks about in [1]_ (also refer to other
    works by Fannjiang - [2]_ and [3]_).

    Thus, to avoid the affine phase ambiguity, we assume that the sample support is *priorly known*, i.e. we impose
    a *tight support*. If we do not remove the affine phase ambiguity this way, then we have to remove the phase
    ramp post-reconstruction, which is more difficult and error-prone.

    For the far-field ptychography simulations, by adding this border, we can additionally ensure overlapping probe
    coverage through the entirety of the sample (including the edges).

    We cannot use this approach for the Bragg ptychography case---while we get transmission through empty
    space, we do not get diffraction (at the Bragg angle) through empty space. As such, the bright field boundary
    condition does not apply.

    References
    ----------
    .. [1] Fannjiang, A. & Zhang, Z. Blind Ptychography by Douglas-Rachford Splitting. 1–21 (2018)
        (http://arxiv.org/abs/1809.00962).
    .. [2] Fannjiang, A. & Chen, P. Blind Ptychography: Uniqueness and Ambiguities. 1–29 (2018)
        (http://arxiv.org/abs/1806.02674).
    .. [3] Fannjiang, A. Raster Grid Pathology and the Cure. Multiscale Modeling & Simulation, 17(3), 973-995 (2019).
    """
    def __init__(self, shape: Tuple[int, ...],
                 border_shape: Tuple[Tuple[int, int], ...] = None,
                 border_const: complex = 0,
                 mod_range: float = None,
                 phase_range: float = None,
                 pixel_size: Tuple[float,...] = None) -> None:
        self.shape = shape
        self.mod_range = mod_range
        self.phase_range = phase_range
        self.pixel_size = pixel_size
        self.border_const = border_const
        if border_shape is not None:
            self._border_shape = border_shape
        else:
            self._border_shape = tuple((0, 0) for i in range(len(shape)))

        self._setObjArrayValues()

    def _setObjArrayValues(self, values: Optional[np.ndarray] = None) -> None:
        """Set the obj transmission function and add the border.

        Performs sanity checks for the 'shape' and '_border_shape' parameters supplied when the class is created. The
        'shape' parameter should be tuple-like and composed of integers, formatted so that 'numpy.zeros' accepts it
        as an argument for the 'shape' parameter. The '_border_shape' should be formatted so that 'numpy.pad' accepts it
        as an argument for the 'pad' parameter.

        Sets the values for the 'array' and 'bordered_array' attributes.
        Parameters
        ----------
        values : array_like, optional
            Obj array values. For the default value 'None', the function creates an array of zeros.
        """

        if values is None:
            try:
                values = np.zeros(self.shape)
            except Exception as e:
                e2 = ValueError("Error in input obj shape.")
                logger.error([e, e2])
                raise e2 from e

        #self._array = values
        try:
            self.bordered_array = np.pad(values, #self._array,
                                         self._border_shape,
                                         mode='constant',
                                         constant_values=self.border_const)
        except Exception as e:
            e2 = ValueError("Error in border specifications.")
            logger.error([e, e2])
            raise e2 from e
        array_slices = tuple(slice(b[0], self.shape[i] + b[0]) for i, b in enumerate(self._border_shape))

        # This is only a view to bordered_array. Changing one changes the other.
        self._array = self.bordered_array[array_slices]


    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, array):
        self._setObjArrayValues(array)

    @property
    def border_shape(self):
        return self._border_shape

    @border_shape.setter
    def border_shape(self, border_shape):
        values = self._array.copy()
        self._border_shape = border_shape
        self._setObjArrayValues(values)


class CustomObjFromArray(Obj):
    r"""Create a `Obj` using a supplied array of values..

    See documentation for `Obj` for information on the attributes.

    Parameters
    ----------
    array: array_like(complex)
        Array that contains the obj values.
    border_shape, border_const, pixel_size : see documentation for `Obj`.

    See also
    --------
    Probe
    """

    def __init__(self, array: np.ndarray,
                 border_shape: Any = 0,
                 border_const: complex = 0,
                 pixel_size: Tuple[float, ...] = None) -> None:
        shape = array.shape
        super().__init__(shape, border_shape, border_const, pixel_size=pixel_size)
        self._setObjArrayValues(array.copy())


class Simulated2DObj(Obj):
    r"""Creates a sample complex-valued obj using stock  data from the skimage library.

    Uses the stock camera image for the phase and the stock immunohistochemistry  image (channel 0) for the
    modulus [4]_.

    See documentation for `Obj` for information on the attributes.

    Parameters
    ----------
    shape : Tuple[int, int]
        Shape of 2d obj array. Default is (128, 128).
    border_shape : array_like
        Default value adds a border of 32 pixels each on the top, bottom, left, and right of the obj array. See
        documentation of `Obj` for information on the border shape.
    border_const : complex
        Default value is 1.0, which corresponds to free space propagation . See documentation of `Obj` for more
        information.
    mod_range : float
        Default value is 1.0. This covers the entire gamut from opaque (0.0) to free space (1.0) propagation.
    phase_range : float
        Default value is :math:`\pi`.

    See also
    --------
    Obj

    References
    ----------
    .. [4] https://scikit-image.org/docs/dev/api/skimage.data.html
    """
    def __init__(self, shape=(128, 128),
                 border_shape=((32, 32), (32, 32)),
                 border_const=1.0,
                 mod_range=1.0,
                 phase_range=np.pi):
        if len(shape) != 2:
            e = ValueError('Supplied shape is not 2d.')
            logger.error(e)
            raise e
        super().__init__(shape, border_shape, border_const, mod_range, phase_range)
        self._createObj()

    def _createObj(self) -> None:
        """Create the obj array using data from skimage, then set the absolute value and phases using the
        specified ranges.
        """
        phase_img = skimage.img_as_float(skimage.data.camera())[::-1, ::-1]
        mod_img = skimage.img_as_float(skimage.data.immunohistochemistry()[:, :, 0])[::-1, ::-1]
        mod = skimage.transform.resize(mod_img, self.shape,
                                       mode='wrap', preserve_range=True)
        phase = skimage.transform.resize(phase_img, self.shape,
                                         mode='wrap', preserve_range=True)

        # Setting the ranges
        phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase)) * self.phase_range
        mod = (mod - np.min(mod)) / (np.max(mod) - np.min(mod)) * self.mod_range

        # Centering the phase at 0.
        phase = np.angle(np.exp(1j * (phase - scipy.stats.circmean(phase))))
        obj = (mod * np.exp(1j * phase)).astype('complex64')
        self._setObjArrayValues(obj)


class Simulated3DCrystalCell(Obj):
    r"""Creates a sample complex-valued crystal cell.

    The output is non-deterministic.

    Parameters
    ----------
    mesh_shape : Tuple[int, int, int]
        Number of points to use for the Delauney mesh generation. Note that this is **NOT** the shape of the final
        generated crystal cell. The final shape is non-deterministic.
    border_shape : array_like
        Default value adds a border of 32 pixels each on the top, bottom, left, and right of the obj array. See
        documentation of `Obj` for information on the border shape.
    border_const : complex
        Default value is 1.0, which corresponds to free space propagation . See documentation of `Obj` for more
        information.

    See also
    --------
    Obj
    utils.generateCrystalCell3D

    Notes
    -----
    * Since the crystal cell is generated randomly at each call of `utils.generateCrystalCell3D`, the output is not
        deterministic.
    * Based on conversations with Marc Allain ( I might have misinterpreted him...), for Bragg ptychography:
        - the border shape should be set so that the sample support is restricted along the line perpendicular to the
            exit wave direction (in the plane  that contains the rocking curve). I think that applying a *known* border
            such that the number of pixels here is :math:`>2\times` the support (or reconstruction
            variable) size in this direction should provide sufficient oversampling.
        - For ptychography, in the probe translation plane, we generally want the object array size to be
            :math:`>2\times` the size of the probe

    """
    def __init__(self, mesh_shape: Tuple[int, int, int] = (128, 128, 128),
                 mod_const: float = 0.5,
                 border_shape=((0,0), (0,0), (0,0)),
                 border_const=0.0,
                 pixel_size=None):
        if len(mesh_shape) != 3:
            e = ValueError('Supplied shape is not 3d.')
            logger.error(e)
            raise e

        self.mod_constant = mod_const

        self.border_const = border_const
        self._border_shape = border_shape
        self.pixel_size = pixel_size

        self.mod_range = None
        self.phase_range = None

        self._createObj()
        self.array *= self.mod_constant

    def _createObj(self) -> None:

        from ptychoSampling.utils.generateCrystalCell3D import generateCrystalCell, trimAndPadCell
        cell = generateCrystalCell(x_points=128, y_points=128, z_points=128)
        cell = trimAndPadCell(cell)
        self.shape = cell.shape
        self._setObjArrayValues(cell)