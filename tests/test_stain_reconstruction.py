import numpy as np
import pytest
from skimage.data import immunohistochemistry
from skimage.metrics import structural_similarity

from rationai.staining import ColorConversion, convert_color
from rationai.staining.typing import RGBArray


IMAGES_AND_CONVS = [
    (
        immunohistochemistry(),
        ColorConversion.RGB2HDR,
    ),
    (
        immunohistochemistry(),
        ColorConversion.RGB2HDR_LEGACY,
    ),
    (
        "sample1_original",
        ColorConversion.RGB2HER,
    ),
]

TEST_IDS = ["skimage-default; H&DAB", "skimage-default; H&DAB_LEGACY", "Sample1; H&E"]


@pytest.mark.parametrize("original,conv", IMAGES_AND_CONVS, ids=TEST_IDS)
def test_structural_similarity(
    original: RGBArray | str, conv: ColorConversion, request: pytest.FixtureRequest
) -> None:
    """For reconstruction, only structural similarity is tested.

    Due to implementation details, the reconsturction is not idempotent,
    and some bigger color shifts can occur in some conversions.
    Therefore, the color similarity cannot be reliably tested.
    """
    if isinstance(original, str):
        original = request.getfixturevalue(original)

    c0, c1, c2 = convert_color(original, conv)  # type: ignore[arg-type]
    reconstructed = convert_color(np.stack([c0, c1, c2], axis=-1), conv.inverse)

    assert structural_similarity(original, reconstructed, channel_axis=-1) > 0.95
