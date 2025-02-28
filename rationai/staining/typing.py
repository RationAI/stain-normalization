from collections.abc import Callable
from typing import TypeAlias

from numpy import float64, uint8
from numpy.typing import NDArray
from PIL.Image import Image


ModifyFunction: TypeAlias = Callable[
    [NDArray[float64], NDArray[float64], NDArray[float64]],
    tuple[NDArray[float64], NDArray[float64], NDArray[float64]],
]
"""
Type definition for function that modifies stain channels.

The function takes three arrays in stain space and returns their modified versions.
"""

Tile: TypeAlias = NDArray[float64 | uint8] | Image
"""
Type definition for a single tile.
"""

StainArray: TypeAlias = NDArray[float64]
"""
Type definition for an array of stain values.
"""

StainChannels: TypeAlias = tuple[StainArray, StainArray, StainArray]
"""
Type definition for all three deconvoluted stain channels.
"""

StainTuple: TypeAlias = tuple[float, float, float]
"""
Type definition for a tuple of stain values.
"""

StainTupleMatrix: TypeAlias = tuple[StainTuple, StainTuple, StainTuple]
"""
Type definition for a stain matrix represented by a tuple of tuples.
"""

RGBArray: TypeAlias = NDArray[uint8]
"""
Type definition for an array of RGB values.
"""
