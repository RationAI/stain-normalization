from typing import Any

import numpy as np
from albumentations import ImageOnlyTransform
from numpy.typing import NDArray
from skimage.util import img_as_float


class ExposureAdjustment(ImageOnlyTransform):  # type: ignore[misc]  # untyped import
    """Adjust the exposure of an image by scaling its brightness.

    Attributes:
        brightness_range: Range specifying the lower and upper bounds for the
            random brightness scaling factor. Values less than 1.0 darken the image,
            while values greater than 1.0 brighten it.
        p: Probability of applying the transformation.
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.brightness_range = brightness_range

    def apply(self, img: NDArray[Any], **params: Any) -> NDArray[Any]:
        """Apply brightness scaling to the image.

        Args:
            img: Input image whose brightness will be adjusted.
            params: Additional parameters.

        Returns:
            RGB image with adjusted brightness as a float32
            NumPy array with values in [0.0, 1.].
        """
        brightness_factor = np.random.uniform(*self.brightness_range)
        img_float = img_as_float(img)
        img = np.clip(img_float * brightness_factor, 0.0, 1.0)
        return img.astype(np.float32)
