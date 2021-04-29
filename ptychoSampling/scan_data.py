import numpy as np
from typing import Any
import dataclasses as dt
from ptychoSampling.logger import logger

@dt.dataclass(frozen=True)
class ScanData:
    """Dataclass that stores the scan data.

    The dataclass is frozen after creation so that the parameters are not changed by accident.

    Note that all the parameters supplied are also the attributes for the dataclass.

    Parameters
    ----------
    wavelength : float
        Wavelength of the scanning beam (in m).
    obj_det_dist : float
        Sample detector distance (in m).
    det_pixel_pitch : float
        Detector pixel pitch (in m). Assumes square detector pixels.
    det_npix : int
        Number of pixels in each side of the detector. Assumes square detector.
    scan_positions : array_like(float)
        Ptychographic scan positions, expressed in terms of probe translations (in m). The stored positions
        correspond to the lower left edge of the probe array, and the origin (:math:`(0,0)` position) corresponds to
        the extreme lower left edge - the coordinates supplied are all positive.
    intensity_patterns : array_like(float)
        Intensities recorded at the various scan positions.
    background_intensity_level : array_like or float
        Background intensities. Can be either a `float` (when the background level is constant throughout),
        a 2d array (when the background intensity differs per pixel, but is constant at every scan position),
        or of the same shape as 'intensity_patterns` (when the background differs at every scan position). Default
        value is `0`.
    full_field_probe : bool
        Whether the probe beam is full-field. In this scenario, for the reconstruction, it can convenient to express
        the ptychography scan in terms of object translations rather than probe translations. Default value is `False`.
    additional_params : dict
        Dictionary containing any additional parameters necessary to describe the scan. The dictionary should
        consist of "key:value" pairs where the "key" is a string for the name of the parameter. The "value" can take
        any form necessary. For convenient access, these individual parameters are then added to the list of
        attributes of the dataclass (but not the *fields* of the dataclass). The parameters supplied should not have
        the same name as any existing class parameters. See Examples.
    memo : str
        String that contains any additional notes for the ptychographic scan. Empty as default.
    Attributes
    ----------
    see Parameters

    Examples
    --------
    For an example for adding and using `additional_params`, we consider a case where the detector is not
    perpendicular to the probe direction, but is offset by some angle :math:`\theta`::

        p = ScanParams(..., additional_params={'detector_angle_offset'=0.2)
        # Both the following statements print output "0.2".
        print(p.additional_params['detector_angle_offset'])
        print(p.detector_angle_offset)

    Note that, however, the dataclass does not include "detector_angle_offset" as a field.

    .. todo::

        Add examples. Also add run_methods to save and load parameters
    """
    wavelength: float
    obj_det_dist: float
    det_pixel_pitch: float
    det_npix: int
    scan_positions: np.ndarray
    intensity_patterns: np.ndarray
    background_intensity_level: Any = 0
    full_field_probe: bool = False
    memo: str = ""
    additional_params: dict = dt.field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            _ = np.array(self.intensity_patterns) + self.background_intensity_level
        except Exception as e:
            e2 = ValueError('Invalid format for background intensity level. Should be compatible to the '
                            + 'supplied intensity_patterns parameter value  through a numpy broadcasting operation.')
            logger.error([e, e2])
            raise e2 from e

        for key in self.additional_params:
            object.__setattr__(self, key, self.additional_params[key])
