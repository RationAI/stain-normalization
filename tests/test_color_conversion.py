import numpy as np
import pytest

from rationai.staining import ColorConversion
from rationai.staining.constants import LIGHT_H, QUPATH_DAB, QUPATH_E, QUPATH_H
from rationai.staining.typing import StainTuple
from rationai.staining.utils import residual


@pytest.mark.parametrize(
    "conversion",
    ColorConversion,
    ids=(conversion.name for conversion in ColorConversion),
)
def test_inverse_conversion(conversion: ColorConversion) -> None:
    original = conversion.matrix
    inverse = conversion.inverse.matrix

    assert np.all(np.isclose(original, np.linalg.inv(inverse)))


STAINS = [
    # Expected values are taken from QuPath
    (QUPATH_H, QUPATH_E, (0.316, -0.598, 0.737)),
    (LIGHT_H, QUPATH_DAB, (0.418, -0.796, 0.437)),
    (QUPATH_H, QUPATH_DAB, (0.633, -0.713, 0.302)),
]


@pytest.mark.parametrize("s0,s1,expected", STAINS, ids=["H&E", "H&DAB", "H&DAB_LEGACY"])
def test_residual(s0: StainTuple, s1: StainTuple, expected: StainTuple) -> None:
    assert np.all(np.isclose(residual(s0, s1), expected, atol=0.001))
