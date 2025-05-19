import numpy as np
from albumentations import ImageOnlyTransform
from numpy.typing import NDArray
from skimage.color import combine_stains, hed_from_rgb, rgb_from_hed, separate_stains


class HEDFactor(ImageOnlyTransform):
    """Adjust the intensity of Hematoxylin and Eosin stains in HED color space.

    Attributes:
        h_intensity_range: Range for the random intensity adjustment factor for the Hematoxylin channel.
        e_intensity_range: Range for the random intensity adjustment factor for the Eosin channel.
        always_apply: Whether this transformation should always be applied.
        p: Probability of applying the transformation.
    """

    def __init__(
        self,
        h_range: tuple[float, float] = (0.8, 1.2),
        e_range: tuple[float, float] = (0.8, 1.2),
        always_apply: bool = True,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.h_range = h_range
        self.e_range = e_range

    def apply(self, img: NDArray, **params) -> NDArray:
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
        d = hed_image[:, :, 2]
        modified_rgb = combine_stains(np.stack((h, e, d), axis=-1), rgb_from_hed)

        return modified_rgb
