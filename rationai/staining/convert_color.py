from enum import Enum, auto, unique
from typing import overload

import numpy as np
from PIL.Image import Image
from skimage.color import combine_stains, separate_stains

from rationai.staining.constants import QUPATH_DAB, QUPATH_E, QUPATH_H
from rationai.staining.typing import (
    RGBArray,
    StainArray,
)
from rationai.staining.utils import inv_mat, residual


@unique
class ConversionType(Enum):
    """Enum representing different types of conversions."""

    STAIN2RGB = auto()
    RGB2STAIN = auto()


@unique
class ColorConversion(Enum):
    """Enum representing different color conversions.

    Value of the enum is a tuple with deconvolution matrix and
    the type of the conversion.

    Note:
        Stain channels are denoted by their first letters
        (H for hematoxylin, D for DAB, ...) and residual channel
        is denoted by the letter R.
    """

    HER2RGB = (
        (QUPATH_H, QUPATH_E, residual(QUPATH_H, QUPATH_E)),
        ConversionType.STAIN2RGB,
    )
    """**Combines** `Hematoxylin`, `Eosin` and `Residual` channels **to RGB**."""

    HDR2RGB = (
        (QUPATH_H, QUPATH_DAB, residual(QUPATH_H, QUPATH_DAB)),
        ConversionType.STAIN2RGB,
    )
    """**Combines** `Hematoxylin`, `DAB` and `Residual` channels **to RGB**."""

    RGB2HER = (inv_mat(HER2RGB[0]), ConversionType.RGB2STAIN)
    """**Separates RGB image** into `Hematoxylin`, `Eosin` and `Residual` channels."""

    RGB2HDR = (inv_mat(HDR2RGB[0]), ConversionType.RGB2STAIN)
    """**Separates RGB image** into `Hematoxylin`, `DAB` and `Residual` channels."""

    @property
    def conv_type(self) -> ConversionType:
        """Returns conversion type associated with the color conversion.

        Returns:
            Conversion type associated with the color conversion.
        """
        return self.value[1]

    @property
    def matrix(self) -> StainArray:
        """Returns the real form of the color deconvolution matrix.

        Returns:
            Deconvolution matrix (3x3) represented by the color conversion
                as a numpy array.
        """
        return np.stack(self.value[0])

    @property
    def inverse(self) -> "ColorConversion":
        """Returns the inverse color conversion.

        Returns:
            Inverse color conversion (i. e., RGB2HER is inverse to HER2RGB).
        """
        match self:
            case ColorConversion.RGB2HER:
                return ColorConversion.HER2RGB
            case ColorConversion.RGB2HDR:
                return ColorConversion.HDR2RGB
            case ColorConversion.HER2RGB:
                return ColorConversion.RGB2HER
            case ColorConversion.HDR2RGB:
                return ColorConversion.RGB2HDR


@overload
def convert_color(
    tile: RGBArray | Image, conversion: ColorConversion
) -> tuple[StainArray, ...]: ...


@overload
def convert_color(tile: StainArray, conversion: ColorConversion) -> RGBArray: ...


def convert_color(
    tile: RGBArray | StainArray | Image, conversion: ColorConversion
) -> RGBArray | tuple[StainArray, ...]:
    """Converts a tile into the specified color space.

    Based on the provided conversion enum, either color deconvolution
    or stain channel combination is performed.

    Note:
        Due to implementation details, it is **NOT GUARANTEED** that
        `convert_tile(convert_tile(I, conv), conv.inverse) == I`. Difference between the
        images can also vary depending on the used conversion.

    Args:
        tile: Tile that should be converted. Can either be a RGB image or three stain
            channels stacked along the last axis. Stain channels can be stacked using
            the following code:
            >>> stacked_stains = np.stack([c0, c1, c2], axis=-1)

            Here, `c[0-2]` are the individual stain channels.

        conversion: Desired color conversion.

    Returns:
        Depending on the conversion, the result can either be a RGB image,
        or a tuple of the separated stain channels.
    """
    match conversion.conv_type:
        case ConversionType.RGB2STAIN:
            return tuple(np.moveaxis(separate_stains(tile, conversion.matrix), -1, 0))

        case ConversionType.STAIN2RGB:
            tile = np.asarray(tile, dtype=np.float64)
            return (255 * combine_stains(tile, conversion.matrix)).astype(np.uint8)
