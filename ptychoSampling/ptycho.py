import numpy as np
import abc
import ptychoSampling.probe
import ptychoSampling.obj
import ptychoSampling.grid
import ptychoSampling.detector
from ptychoSampling.utils import utils
from ptychoSampling.logger import logger


class Ptychography(abc.ABC):
    """This class is incomplete right now."""
    def __init__(self, probe: ptychoSampling.probe.Probe,
                 obj: ptychoSampling.obj.Obj,
                 scan_grid: ptychoSampling.grid.ScanGrid,
                 detector: ptychoSampling.detector.Detector,
                 upsampling_factor : int = 1):

        self._probe = probe
        self._obj = obj
        self.upsampling_factor = upsampling_factor
        self._detector = detector
        self._scan_grid = scan_grid

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, obj_in: ptychoSampling.obj.Obj):
        self._obj = obj_in
        self._sanityChecks()

    @property
    def probe(self):
        return self._probe

    @probe.setter
    def probe(self, probe_in: ptychoSampling.probe.Probe):
        self._probe = probe_in
        self._sanityChecks()

    def _sanityChecks(self):
        sanity_checks = {'probe_det': self._probe.npix == self._detector.npix * self.upsampling_factor,
                         'probe_obj': self._probe.pixel_size == self._obj.pixel_size,
                         'obj_grid': self._obj.pixel_size == self._scan_grid.obj_pixel_size,
                         'probe_grid': self._probe.npix == self._scan_grid.probe_npix}

        for s, v in sanity_checks.items():
            if not v:
                e = ValueError(f"Mismatch in supplied {s} parameters.")
                logger.error(e)
                raise e

    @abc.abstractmethod
    def _getObjPixelSize(self) -> float:
        pass

    @abc.abstractmethod
    def getProbeFromDiffractionIntensities(self, threshold_zero: float = 0,
                                           threshold_max: float = 1.0) -> ptychoSampling.probe.Probe:
        pass

    def getObjFromScanPositions(self,
                                 border_npix : int = 0,
                                 border_fill_const: complex = 0,
                                 obj_fill_const : complex = None,
                                 mod_range: float = 1.0,
                                 phase_range: float = np.pi) -> ptychoSampling.obj.Obj:
        pass


    @classmethod
    @abc.abstractmethod
    def createFromScanData(cls, scan_data: ptychoSampling.scan_data.ScanData) -> 'Ptychography':
        pass

    @abc.abstractmethod
    def createReconstructor(self):
        pass

    @staticmethod
    def _checkGuessNpix(args_dict: dict,
                        guess: np.ndarray) -> dict:
        """ I need to check to make sure that this method is needed/useful at all.

        Ensure that the number of pixels is supplied for the object and probe inits."""
        if guess is not None:
            print(f'Supplied guess overrides any npix value in {args_dict}.')
            args_dict['npix'] = guess.shape[0]
        elif ('npix' not in args_dict):
            raise KeyError(f'Need to supply either the guess or npix (in {args_dict}).')
        return args_dict


class FarfieldPtychography(Ptychography):

    def initProbeFromDiffractionData(self, threshold_zero: float = 0,
                                     threshold_max: float = 1.0) -> None:
        intensities = self.params.intensity_patterns - self.params.background_intensity_level
        max_intensity = np.max(intensities)
        intensities[intensities < threshold_zero * max_intensity] = 0
        intensities[intensities > threshold_max * max_intensity] = threshold_max * max_intensity
        intensity_avg = np.mean(intensities, axis=0)
        if self.upsample_factor > 1:
            intensity_avg = utils.upSample(intensity_avg, self.upsample_factor)

        detector_wavefront = Wavefront(array=np.sqrt(intensity_avg).astype('complex64'))
        prb_guess = propFF(detector_wavefront, apply_phase_factor=False, backward=True)
        self.probe = ptychoSampling.probe.CustomProbeFromArray(wavefront_array=prb_guess.array,
                                          wavelength=self.params.wavelength,
                                          pixel_size=self.obj_pixel_size)

    def initObjFromScanPositions(self,
                                 border_npix : int = 0,
                                 border_fill_const: complex = 0,
                                 obj_fill_const : complex = None,
                                 mod_range: float = 1.0,
                                 phase_range: float = np.pi) -> None:
        max_x_position = self.positions_pix[:, 0]
        max_y_position = self.positions_pix[:, 1]
        obj_x = max_x_position + self.params.det_npix * self.upsample_factor
        obj_y = max_y_position + self.params.det_npix * self.upsample_factor

        if obj_fill_const is None:
            obj_vals = utils.getRandomComplexArray(shape=(obj_y, obj_x), mod_range=mod_range, phase_range=phase_range)
        else:
            obj_vals = np.ones((obj_y, obj_x)) * obj_fill_const
        self.obj = ptychoSampling.obj.CustomObjFromArray(array=obj_vals,
                                                 border_shape=border_npix,
                                                 border_const=border_fill_const,
                                                 pixel_size=self.obj_pixel_size)


