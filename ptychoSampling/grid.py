from typing import Tuple
import numpy as np
from ptychoSampling.logger import  logger
import abc

__all__ = ["ScanGrid",
           "BraggPtychoGrid",
           "CustomGridFromPositionDistances",
           "CustomGridFromPositionPixels",
           "RectangleGrid"]

class ScanGrid(abc.ABC):
    def __init__(self, obj_w_border_shape: Tuple[int, int],
                 probe_shape: Tuple[int, int],
                 obj_pixel_size: Tuple[float, float],
                 step_dist: Tuple[float, float],
                 subpixel_scan: bool = False,
                 scan_grid_boundary_pix: np.ndarray = None,
                 full_field_probe: bool = False):
        self.obj_w_border_shape = obj_w_border_shape
        self.obj_pixel_size = obj_pixel_size
        self.probe_shape = probe_shape
        self.step_dist = np.array(step_dist)
        self.subpixel_scan = subpixel_scan
        self.full_field_probe = full_field_probe

        if scan_grid_boundary_pix is None:
            if not self.full_field_probe:
                self.scan_grid_boundary_pix = np.array([[0,self.obj_w_border_shape[0]],
                                                        [0, self.obj_w_border_shape[1]]])
            else:
                self.scan_grid_boundary_pix = np.array([[0, probe_shape[0]], [0, probe_shape[1]]])
        else:
            try:
                _ = np.reshape(scan_grid_boundary_pix, (2,2))
            except Exception as e:
                e2 = ValueError("scan_grid_boundary_pix should contain integers and have shape [[y_min, y_max], "
                                + "[x_min, x_max]]")
                logger.error([e, e2])
                raise e2 from e
            self.scan_grid_boundary_pix = scan_grid_boundary_pix

        self.positions_pix = []
        self.positions_subpix = []
        self.positions_dist = []

    def checkOverlap(self, overlap_ratio:float = 0.5) -> bool:
        if not self.full_field_probe:
            overlap_nx = overlap_ratio * self.probe_shape[1]
            overlap_ny = overlap_ratio * self.probe_shape[0]
        else:
            overlap_nx = overlap_ratio * self.obj_w_border_shape[1]
            overlap_ny = overlap_ratio * self.obj_w_border_shape[0]

        for p in self.positions_pix:
            differences = np.abs(self.positions_pix - p)
            ydiffs = differences[:,0]
            xdiffs = differences[:,1]
            xmin = np.min(xdiffs[xdiffs > 0])
            ymin = np.min(ydiffs[ydiffs > 0])

            if xmin > overlap_nx or ymin > overlap_ny:
                e = ValueError("Insufficient overlap between adjacent scan positions.")
                logger.error(e)
                raise e
        return True

class RectangleGrid(ScanGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calculatePositions()

    def _calculatePositions(self):
        psy, psx = self.obj_pixel_size
        sy, sx = self.step_dist

        xmin = np.max((0, self.scan_grid_boundary_pix[1][0]))
        ymin = np.max((0, self.scan_grid_boundary_pix[0][0]))
        if not self.full_field_probe:
            xmax = np.min((self.obj_w_border_shape[1], self.scan_grid_boundary_pix[1][1])) - self.probe_shape[1]
            ymax = np.min((self.obj_w_border_shape[0], self.scan_grid_boundary_pix[0][1])) - self.probe_shape[0]
        else:
            xmax = np.min((self.probe_shape[1], self.scan_grid_boundary_pix[1][1])) - self.obj_w_border_shape[1]
            ymax = np.min((self.probe_shape[0], self.scan_grid_boundary_pix[0][1])) - self.obj_w_border_shape[0]

        x_positions = np.arange(xmin, xmax, sx / psx)
        y_positions = np.arange(ymin, ymax, sy / psy)

        positions = np.array([(y, x) for y in y_positions for x in x_positions])

        self.positions_pix = (positions // 1).astype('int')
        self.positions_subpix = positions % 1 if self.subpixel_scan else np.zeros_like(positions)
        self.positions_dist = positions * np.array([sy, sx])

class CustomGridFromPositionDistances(ScanGrid):
    def __init__(self, obj_pixel_size: Tuple[float, float],
                 positions_dist: np.ndarray,
                 obj_w_border_shape: Tuple[int, int] = None,
                 probe_shape: Tuple[int, int] = None,
                 step_dist: Tuple[float, float] = None,
                 subpixel_scan: bool = False,
                 scan_grid_boundary_npix: np.ndarray = None) -> None:

        super().__init__(obj_w_border_shape,
                         probe_shape,
                         obj_pixel_size,
                         step_dist,
                         subpixel_scan,
                         scan_grid_boundary_npix)

        self.positions_dist = positions_dist
        positions = self.positions_dist / self.obj_pixel_size
        self.positions_pix = (positions // 1).astype('int')
        self.positions_subpix = (positions % 1) if self.subpixel_scan else np.zeros_like(positions)

class CustomGridFromPositionPixels(ScanGrid):
    def __init__(self, obj_pixel_size: Tuple[float, float],
                 positions_pix: np.ndarray,
                 positions_subpix: np.ndarray = None,
                 obj_w_border_shape: Tuple[int, int] = None,
                 probe_shape: Tuple[int, int] = None,
                 step_dist: Tuple[float, float] = None,
                 scan_grid_boundary_pix: np.ndarray = None) -> None:

        subpixel_scan = True if positions_subpix is not None else False
        super().__init__(obj_w_border_shape,
                        probe_shape,
                        obj_pixel_size,
                        step_dist,
                        subpixel_scan,
                        scan_grid_boundary_pix)

        self.positions_pix = positions_pix
        self.positions_subpix = np.zeros(positions_pix.shape) if positions_subpix is None else positions_subpix
        self.positions_dist = (self.positions_pix + self.positions_subpix) * self.obj_pixel_size


class BraggPtychoGrid(ScanGrid):
    def __init__(self, *args: int,
                 positions_pix: np.ndarray,
                 positions_subpix: np.ndarray,
                 positions_dist: np.ndarray,
                 grid2d_axes: Tuple[str, str],
                 two_theta_deg: float,
                 del_omega_deg: float,
                 n_rc_angles: int,
                 **kwargs: int):
        super().__init__(*args, **kwargs)
        self.positions_pix = positions_pix
        self.positions_subpix = positions_subpix
        self.positions_dist = positions_dist
        self.grid2d_axes = grid2d_axes
        self._initAngleGrid(two_theta_deg, del_omega_deg, n_rc_angles)


    def _initAngleGrid(self, two_theta: float,
                       del_omega: float,
                       n_rc_angles: int,):
        self.two_theta = two_theta
        self.del_omega = del_omega
        self.n_rc_angles = n_rc_angles

        # To account for odd and even number of rc angles
        self.rc_angles = np.arange(-(n_rc_angles // 2), n_rc_angles // 2 + n_rc_angles % 2) * self.del_omega

        full_rc_positions_indices = []
        for a in range(self.n_rc_angles):
            for p in range(self.positions_pix.shape[0]):
                full_rc_positions_indices.append([a, p])
        self.full_rc_positions_indices = np.array(full_rc_positions_indices)

    @classmethod
    def fromPtychoScan2D(cls, grid2d: ScanGrid,
                         grid2d_axes: Tuple[str, str],
                         two_theta: float,
                         del_omega: float,
                         n_rc_angles: int):
        ob = cls.__new__(cls)
        ob.__dict__ = grid2d.__dict__.copy()
        ob.grid2d_axes = grid2d_axes
        ob._initAngleGrid(two_theta, del_omega, n_rc_angles)
        return ob