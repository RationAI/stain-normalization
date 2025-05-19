import numpy as np
from albumentations import ImageOnlyTransform
from numpy.typing import NDArray


class ExposureAdjustment(ImageOnlyTransform):
    """Adjust the exposure of an image by scaling its brightness.

    Attributes:
        brightness_range: Range specifying the lower and upper bounds for the
            random brightness scaling factor. Values less than 1.0 darken the image,
            while values greater than 1.0 brighten it.
        always_apply: Whether this transformation should always be applied.
        p: Probability of applying the transformation.
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        always_apply: bool = True,
        p: float = 1,
    ):
        super().__init__(always_apply, p)
        self.brightness_range = brightness_range

    def apply(self, img: NDArray, **params) -> NDArray:
        """Apply brightness scaling to the image.

        Args:
            img: Input image whose brightness will be adjusted.
            params: Additional parameters.

        Returns:
            RGB image with adjusted brightness as a float32 
            NumPy array with values in [0.0, 1.].
        """
        brightness_factor = np.random.uniform(*self.brightness_range)
        img = img.astype(np.float32)
        img = np.clip(img * brightness_factor, 0.0, 1.0)

        return img
