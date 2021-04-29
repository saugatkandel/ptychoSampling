from typing import Tuple
import dataclasses as dt

__all__ = ["Detector"]

@dt.dataclass(frozen=True)
class Detector:
    shape : Tuple[int, int]
    pixel_size : Tuple[float, float]
    obj_dist : float