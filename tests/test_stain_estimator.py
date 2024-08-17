from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from skimage.color import deltaE_ciede2000
from skimage.data import immunohistochemistry
from skimage.filters import gaussian

from rationai.staining import ColorConversion, convert_color, estimate_stain_vectors
from rationai.staining.constants import QUPATH_DAB, QUPATH_E, QUPATH_H
from rationai.staining.typing import StainArray, Tile


ZEROS = np.zeros(shape=(512, 512), dtype=np.float64)


H_RGB = np.array([88, 76, 180], dtype=np.uint8)  # from QuPath
E_RGB = np.array([199, 50, 112], dtype=np.uint8)  # from QuPath
DAB_RGB = np.array([186, 110, 56], dtype=np.uint8)  # from QuPath


NANS = np.full(shape=(3), fill_value=np.nan)

GENERATED_DIR = Path("tests/data/generated")


def _blur(img: NDArray[np.uint8], sigma: float) -> NDArray[np.uint8]:
    result = np.empty_like(img)
    channels = [img[..., 0], img[..., 1], img[..., 2]]

    for i, c in enumerate(channels):
        result[..., i] = (255 * gaussian(c, sigma=sigma)).astype(np.uint8)

    return result


# Input, first expected stain, second expected stain
DATA = [
    (
        immunohistochemistry(),
        QUPATH_H,
        QUPATH_DAB,
    ),
    (
        np.asarray(Image.open(GENERATED_DIR / "h_e.jpg")),
        QUPATH_H,
        QUPATH_E,
    ),
    (
        np.asarray(Image.open(GENERATED_DIR / "h_dab.jpg")),
        QUPATH_H,
        QUPATH_DAB,
    ),
    (
        _blur(np.asarray(Image.open(GENERATED_DIR / "h_e.jpg")), sigma=5),
        QUPATH_H,
        QUPATH_E,
    ),
    (
        _blur(np.asarray(Image.open(GENERATED_DIR / "h_dab.jpg")), sigma=5),
        QUPATH_H,
        QUPATH_DAB,
    ),
    (
        convert_color(np.stack([ZEROS] * 3, axis=-1), ColorConversion.HER2RGB),
        NANS,
        NANS,
    ),
]

IDS = [
    "Default skimage image H&DAB",
    "Artificial H&E",
    "Artificial H&DAB",
    "Blurred Artificial H&E",
    "Blurred Artificial H&DAB",
    "Empty Image",
]


@pytest.mark.parametrize("img,expected1,expected2", DATA, ids=IDS)
class TestStainEstimator:
    @staticmethod
    def _both_nan(x: NDArray[Any], y: NDArray[Any]) -> bool:
        return bool(np.isnan(x).all()) and bool(np.isnan(y).all())

    def test_estimation_vector_similarity(
        self, img: Tile, expected1: StainArray, expected2: StainArray
    ) -> None:
        stain1, stain2 = estimate_stain_vectors(img)

        assert np.dot(stain1, expected1) > 0.97 or self._both_nan(stain1, expected1)
        assert np.dot(stain2, expected2) > 0.97 or self._both_nan(stain2, expected2)

    def test_estimation_color_difference(
        self, img: Tile, expected1: StainArray, expected2: StainArray
    ) -> None:
        stain1, stain2 = estimate_stain_vectors(img)
        delta = deltaE_ciede2000

        assert delta(stain1, expected1) < 1 or self._both_nan(stain1, expected1)
        assert delta(stain2, expected2) < 1 or self._both_nan(stain2, expected2)
