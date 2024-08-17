import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from rationai.staining.typing import RGBArray


@pytest.fixture
def white() -> RGBArray:
    """Returns an array with all-white RBG image of size (512, 512)."""
    return np.full(shape=(512, 512, 3), dtype=np.uint8, fill_value=255)


@pytest.fixture
def black() -> RGBArray:
    """Returns an array with all-black RBG image of size (512, 512)."""
    return np.full(shape=(512, 512, 3), dtype=np.uint8, fill_value=0)


@pytest.fixture
def sample1_original() -> RGBArray:
    """Returns original of the sample1 image."""
    return np.asarray(Image.open("tests/data/sample1/original.jpg"))


@pytest.fixture
def sample1_hematoxylin() -> NDArray[np.uint8]:
    """Returns hematoxylin channel of the sample1 image (created by QuPath)."""
    return np.asarray(Image.open("tests/data/sample1/hematoxylin.png"))


@pytest.fixture
def sample1_eosin() -> NDArray[np.uint8]:
    """Returns eosin channel of the sample1 image (created by QuPath)."""
    return np.asarray(Image.open("tests/data/sample1/eosin.png"))


@pytest.fixture
def sample1_residual() -> NDArray[np.uint8]:
    """Returns residual channel of the sample1 image (created by QuPath)."""
    return np.asarray(Image.open("tests/data/sample1/residual.png"))
