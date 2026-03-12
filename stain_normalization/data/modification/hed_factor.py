from typing import Any

import numpy as np
from albumentations import ImageOnlyTransform
from numpy.typing import NDArray
from skimage.color import combine_stains, hed_from_rgb, rgb_from_hed, separate_stains


class HEDFactor(ImageOnlyTransform):  # type: ignore[misc]  # untyped import
    """Adjust the intensity of Hematoxylin and Eosin stains in HED color space.

    Attributes:
        h_range: Range for the random intensity adjustment factor for the Hematoxylin channel.
        e_range: Range for the random intensity adjustment factor for the Eosin channel.
        p: Probability of applying the transformation.
    """

    def __init__(
        self,
        h_range: tuple[float, float] = (0.8, 1.2),
        e_range: tuple[float, float] = (0.8, 1.2),
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.h_range = h_range
        self.e_range = e_range

    def apply(self, img: NDArray[Any], **params: Any) -> NDArray[Any]:
        """Apply the modification to the image.

        Args:
            img: Image to which the transformation will be applied.
            params: Additional parameters.

        Returns:
            RGB image with modified Hematoxylin and Eosin channels
            as a float32 NumPy array with values in [0.0, 1.].
        """
        h_factor = np.random.uniform(*self.h_range)
        e_factor = np.random.uniform(*self.e_range)

        hed_image = separate_stains(img, hed_from_rgb)
        h = hed_image[:, :, 0] * h_factor
        e = hed_image[:, :, 1] * e_factor
        d = hed_image[:, :, 2]  # DAB channel unchanged
        modified_rgb = combine_stains(np.stack((h, e, d), axis=-1), rgb_from_hed)

        return np.clip(modified_rgb, 0, 1).astype(np.float32)
