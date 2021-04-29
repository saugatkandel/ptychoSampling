import numpy as np
import dataclasses as dt
from typing import Tuple, Any
from ptychoSampling.obj import Simulated2DObj, Obj
from ptychoSampling.probe import Probe, FocusCircularProbe
from ptychoSampling.grid import RectangleGrid, ScanGrid, CustomGridFromPositionPixels
from ptychoSampling.detector import Detector
from ptychoSampling.logger import logger

@dt.dataclass
class ObjParams:
    shape: Tuple = (160, 160) # y, x
    border_shape: Any = ((32, 32),(32, 32))
    border_const: float = 1.0

@dt.dataclass
class ProbeParams:
    n_photons: float = 1e4
    defocus_dist: float = 0.05
    width_dist: Tuple = (5e-6, 5e-6) # y, x
    apodize: bool = False

@dt.dataclass
class GridParams:
    step_dist: Tuple[float, float] = (3.0e-6, 3.0e-6) # y, x
    #random_shift_dist: Tuple[float, float] = None
    #random_shift_pix: Tuple[int, int] = None
    #random_shift_ratio: Tuple[float, float] = None

@dt.dataclass
class DetectorParams:
    shape: Tuple[int,int] = (64, 64)
    obj_dist: float = 14.0
    pixel_size: Tuple[float, float] = (55e-6, 55e-6) # y, x


class Simulation:
    def __init__(self,
                 wavelength: float = 1.5e-10,
                 obj: Obj = None,
                 obj_params: dict = None,
                 probe: Probe = None,
                 probe_params: dict = None,
                 scan_grid: ScanGrid = None,
                 scan_grid_params: dict = None,
                 detector: Detector = None,
                 detector_params: dict = None,
                 poisson_noise: bool = True,
                 random_shift_pix: Tuple[int, int] = None,
                 upsampling_factor: int = 1,
                 background_scaling: float=1e-8,
                 background_constant: float=0) -> None:

        self.wavelength = wavelength
        self.upsampling_factor = upsampling_factor
        self.poisson_noise = poisson_noise

        self.background_scaling = background_scaling
        self.background_constant = background_constant

        if obj or probe or scan_grid or detector:
            logger.warning("If one (or all) of obj, probe, scan_grid, or detector is supplied, "
                           + "then the corresponding _params parameter is ignored.")

        if detector is not None:
            self.detector = detector
            self._detector_params = {}
        else:
            detector_params = {} if detector_params is None else detector_params
            self._detector_params = DetectorParams(**detector_params)
            self.detector = Detector(**dt.asdict(self._detector_params))

        detector_support_size = np.asarray(self.detector.pixel_size) * self.detector.shape
        obj_pixel_size = self.wavelength * self.detector.obj_dist / (detector_support_size * self.upsampling_factor)

        probe_shape = np.array(self.detector.shape) * self.upsampling_factor

        if obj is not None:
            if obj.pixel_size is not None and np.any(obj.pixel_size != obj_pixel_size):
                e = ValueError("Mismatch between the provided pixel size and the pixel size calculated from scan "
                               + "parameters.")
                logger.error(e)
                raise e
            obj.pixel_size = obj_pixel_size
        else:
            obj_params = {} if obj_params is None else obj_params
            self._obj_params = ObjParams(**obj_params)
            self.obj = Simulated2DObj(**dt.asdict(self._obj_params))

        if probe is not None:
            check = (np.any(probe.shape != probe_shape)
                     or (probe.wavelength != self.wavelength)
                     or np.any(probe.pixel_size != obj_pixel_size))
            if check:
                e = ValueError("Supplied probe parameters do not match with supplied scan and detector parameters.")
                logger.error(e)
                raise e
            self.probe = probe
        else:
            probe_params = {} if probe_params is None else probe_params
            self._probe_params = ProbeParams(**probe_params)
            self.probe = FocusCircularProbe(wavelength=wavelength,
                                        pixel_size=obj_pixel_size,
                                        shape=probe_shape,
                                        **dt.asdict(self._probe_params))

        if scan_grid is not None:
            self.scan_grid = scan_grid
            self._scan_grid_params = None
        else:
            scan_grid_params = {} if scan_grid_params is None else scan_grid_params
            self._scan_grid_params = GridParams(**scan_grid_params)
            self.scan_grid = RectangleGrid(obj_w_border_shape=self.obj.bordered_array.shape,
                                           probe_shape=self.probe.shape,
                                           obj_pixel_size=obj_pixel_size,
                                           **dt.asdict(self._scan_grid_params))
            #if self._scan_grid_params.random_shift_dist is not None:
            #    raise
            if random_shift_pix is not None:
                self.scan_grid = self._updateGridWithRandomPixShifts(self.scan_grid, random_shift_pix)
            #elif self._scan_grid_params.random_shift_ratio is not None:
            #    raise
        self.scan_grid.checkOverlap()
        self._calculateDiffractionPatterns()

    @staticmethod
    def _updateGridWithRandomPixShifts(scan_grid, random_shift_pix):
        positions = scan_grid.positions_pix
        ymin = np.min(positions[:,0])
        ymax = np.max(positions[:,0])
        xmin = np.min(positions[:,1])
        xmax = np.max(positions[:,1])

        rys  = np.random.randint(-random_shift_pix[0], random_shift_pix[0] + 1, len(positions))
        rxs  = np.random.randint(-random_shift_pix[1], random_shift_pix[1] + 1, len(positions))

        positions_new = []
        for i, (py, px) in enumerate(positions):
            py_new = min(ymax, max(ymin, py + rys[i]))
            px_new = min(xmax, max(xmin, px + rxs[i]))
            positions_new.append([py_new, px_new])

        grid_new = CustomGridFromPositionPixels(obj_pixel_size = scan_grid.obj_pixel_size,
                                                positions_pix = np.array(positions_new),
                                                positions_subpix = None,
                                                obj_w_border_shape= scan_grid.obj_w_border_shape,
                                                probe_shape = scan_grid.probe_shape,
                                                step_dist = scan_grid.step_dist,
                                                scan_grid_boundary_pix=scan_grid.scan_grid_boundary_pix)
        return grid_new


    def _calculateDiffractionPatterns(self):
        wv = self.probe.wavefront.copy()#.array
        intensities_all = []

        for i, (py, px) in enumerate(self.scan_grid.positions_pix):
            if self.scan_grid.subpixel_scan:
                # There might be issues with fft shifting for the subpixel scan
                uy = np.fft.fftshift(np.arange(self.probe.shape[0]))
                ux = np.fft.fftshift(np.arange(self.probe.shape[1]))
                spy, spx = self.scan_grid.subpixel_scan[i]
                phase_factor = (-2 * np.pi * (ux * spx + uy[:,None] * spy) / self.probe.shape[0])
                phase_ramp = np.complex(np.cos(phase_factor), np.sin(phase_factor))
                wv = (self.probe.wavefront.fft2() * phase_ramp).ifft2()
                #wv = np.fft.ifft2(np.fft.fft2(wv, norm='ortho') * phase_ramp, norm='ortho')

            obj_slice = np.fft.fftshift(self.obj.bordered_array[py: py + self.probe.shape[0],
                                        px: px + self.probe.shape[0]])
            wv_out = wv * obj_slice
            #det_wave = np.fft.fft2(wv, norm='ortho')
            #intensity = np.abs(det_wave)**2
            intensities = wv_out.fft2().intensities
            intensities_all.append(intensities)

        self.background_level = self.background_constant + self.background_scaling * np.max(intensities_all)
        self.intensities = np.array(intensities_all) + self.background_level
        if self.poisson_noise:
            self.intensities = np.random.poisson(self.intensities)









