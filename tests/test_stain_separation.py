import numpy as np
import pytest
from numpy.typing import NDArray
from skimage.metrics import structural_similarity

from rationai.staining import ColorConversion, convert_color
from rationai.staining.typing import RGBArray, StainArray


RGB_CONVERSIONS = [
    ColorConversion.RGB2HER,
    ColorConversion.RGB2HDR,
]
RGB_NAMES = [c.name for c in RGB_CONVERSIONS]


@pytest.mark.parametrize("conversion", RGB_CONVERSIONS, ids=RGB_NAMES)
def test_black_image(black: RGBArray, conversion: ColorConversion) -> None:
    c0, c1, c2 = convert_color(black, conversion)

    rgb2stain = np.stack(conversion.value[0])
    expected_values = np.sum(rgb2stain, axis=0)

    assert np.all(c0 == expected_values[0])
    assert np.all(c1 == expected_values[1])
    assert np.all(c2 == expected_values[2])


@pytest.mark.parametrize("conversion", RGB_CONVERSIONS, ids=RGB_NAMES)
def test_white_image(white: RGBArray, conversion: ColorConversion) -> None:
    c0, c1, c2 = convert_color(white, conversion)

    assert np.all(c0 == 0)
    assert np.all(c1 == 0)
    assert np.all(c2 == 0)


def test_real_sample1_image(
    sample1_original: RGBArray,
    sample1_hematoxylin: StainArray,
    sample1_eosin: StainArray,
    sample1_residual: StainArray,
) -> None:
    h, e, r = convert_color(sample1_original, ColorConversion.RGB2HER)

    assert structural_similarity(_prepare(h), sample1_hematoxylin) > 0.95
    assert structural_similarity(_prepare(e), sample1_eosin) > 0.95
    assert structural_similarity(_prepare(r), sample1_residual) > 0.95


def _prepare(c: NDArray[np.float64]) -> NDArray[np.uint8]:
    return (255 * (c / np.max(c))).astype(np.uint8)
