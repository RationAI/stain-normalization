from collections.abc import Sequence
from enum import Enum, auto, unique
from typing import overload

import numpy as np
from PIL.Image import Image
from skimage.color import combine_stains

from rationai.staining.constants import LIGHT_H, QUPATH_DAB, QUPATH_E, QUPATH_H
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
        (LIGHT_H, QUPATH_DAB, residual(LIGHT_H, QUPATH_DAB)),
        ConversionType.STAIN2RGB,
    )
    """**Combines** `Hematoxylin`, `DAB` and `Residual` channels **to RGB**."""

    HDR2RGB_LEGACY = (
        (QUPATH_H, QUPATH_DAB, residual(QUPATH_H, QUPATH_DAB)),
        ConversionType.STAIN2RGB,
    )
    """**Combines** `Hematoxylin`, `DAB` and `Residual` channels **to RGB**.
    
    This conversion uses the Hematoxylin vector from QuPath, and it is no longer
    recommended and is marked as legacy as it did not
    provide expected results for staining detection and separation.
    """

    RGB2HDR = (inv_mat(HDR2RGB[0]), ConversionType.RGB2STAIN)
    """**Separates RGB image** into `Hematoxylin`, `DAB` and `Residual` channels."""

    RGB2HER = (inv_mat(HER2RGB[0]), ConversionType.RGB2STAIN)
    """**Separates RGB image** into `Hematoxylin`, `Eosin` and `Residual` channels."""

    RGB2HDR_LEGACY = (inv_mat(HDR2RGB_LEGACY[0]), ConversionType.RGB2STAIN)
    """**Separates RGB image** into `Hematoxylin`, `DAB` and `Residual` channels.
    
    This conversion uses the Hematoxylin vector from QuPath, and it is no longer
    recommended and is marked as legacy as it did not
    provide expected results for staining detection and separation.
    """

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
            # H&E Protocol
            case ColorConversion.RGB2HER:
                return ColorConversion.HER2RGB
            case ColorConversion.HER2RGB:
                return ColorConversion.RGB2HER

            # H&DAB Protocol
            case ColorConversion.RGB2HDR:
                return ColorConversion.HDR2RGB
            case ColorConversion.HDR2RGB:
                return ColorConversion.RGB2HDR

            # Legacy H&DAB Protocol
            case ColorConversion.RGB2HDR_LEGACY:
                return ColorConversion.HDR2RGB_LEGACY
            case ColorConversion.HDR2RGB_LEGACY:
                return ColorConversion.RGB2HDR_LEGACY


def _separate_stains(
    img: RGBArray | Image, rgb2stain: StainArray, keep_negative_values: bool = False
) -> StainArray:
    values = np.maximum(np.asarray(img).astype(np.float64) / 255, 1e-6)
    stains = (np.log(values) / np.log(1e-6)) @ rgb2stain

    if not keep_negative_values:
        stains = np.maximum(stains, 0)

    return stains


@overload
def convert_color(
    tile: RGBArray | Image,
    conversion: ColorConversion,
    keep_negative_values: bool = False,
) -> tuple[StainArray, ...]: ...


@overload
def convert_color(
    tile: StainArray | Sequence[StainArray],
    conversion: ColorConversion,
    keep_negative_values: bool = False,
) -> RGBArray: ...


def convert_color(
    tile: RGBArray | Image | StainArray | Sequence[StainArray],
    conversion: ColorConversion,
    keep_negative_values: bool = False,
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
            ```python
            stacked_stains = np.stack([c0, c1, c2], axis=-1)
            ```
            Here, `c[0-2]` are the individual stain channels.

        conversion: Desired color conversion.

        keep_negative_values: Decides if the negative values produced by color separation
            should be kept.

    Returns:
        Depending on the conversion, the result can either be a RGB image,
        or a tuple of the separated stain channels.
    """
    match conversion.conv_type:
        case ConversionType.RGB2STAIN:
            return tuple(
                np.moveaxis(
                    _separate_stains(tile, conversion.matrix, keep_negative_values),  # type: ignore[arg-type]
                    -1,
                    0,
                )
            )

        case ConversionType.STAIN2RGB:
            if isinstance(tile, Sequence):
                tile = np.stack(tile, axis=-1)

            tile = np.asarray(tile, dtype=np.float64)
            return (255 * combine_stains(tile, conversion.matrix)).astype(np.uint8)
