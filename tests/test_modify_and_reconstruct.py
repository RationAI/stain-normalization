import numpy as np
import pytest
from numpy.typing import NDArray
from skimage.data import immunohistochemistry
from skimage.metrics import structural_similarity

from rationai.staining import modify_and_reconstruct
from rationai.staining.constants import LIGHT_H, QUPATH_DAB, QUPATH_E, QUPATH_H
from rationai.staining.typing import ModifyFunction, RGBArray, StainTuple


C = NDArray[np.float64]


def _no_change(c0: C, c1: C, c2: C) -> tuple[C, C, C]:
    return c0, c1, c2


IMAGES_AND_STAINS = [
    (
        immunohistochemistry(),
        _no_change,
        LIGHT_H,
        QUPATH_DAB,
        None,
    ),
    (
        "sample1_original",
        _no_change,
        QUPATH_H,
        QUPATH_E,
        None,
    ),
]

TEST_IDS = ["skimage-default; no change", "Sample1; no change"]


@pytest.mark.parametrize(
    "original,modify,stain0,stain1,stain2", IMAGES_AND_STAINS, ids=TEST_IDS
)
def test_modify_and_reconstruct(
    original: RGBArray | str,
    modify: ModifyFunction,
    stain0: StainTuple,
    stain1: StainTuple,
    stain2: StainTuple,
    request: pytest.FixtureRequest,
) -> None:
    if isinstance(original, str):
        original = request.getfixturevalue(original)

    reconstructed = modify_and_reconstruct(
        tile=original,  # type: ignore[arg-type]
        modify=modify,
        stain0=stain0,
        stain1=stain1,
        stain2=stain2,
    )

    assert structural_similarity(original, reconstructed, channel_axis=-1) > 0.99
