import numpy as np
from numpy import float64

from rationai.staining.typing import StainTuple
from rationai.staining.utils.normalize import normalize


def residual(c0: StainTuple, c1: StainTuple) -> StainTuple:
    c0_arr = np.array(c0, dtype=float64)
    c1_arr = np.array(c1, dtype=float64)

    return tuple(np.round(normalize(np.cross(c0_arr, c1_arr)), decimals=3))
