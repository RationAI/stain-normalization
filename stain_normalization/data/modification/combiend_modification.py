import numpy as np
from albumentations import ImageOnlyTransform
from numpy.typing import NDArray
from skimage import exposure
from skimage.color import combine_stains, hed_from_rgb, rgb_from_hed, separate_stains


class CombinedModifications(ImageOnlyTransform):
    """Apply combined modifications to the H&E channels in HED color space.

    Attributes:
        intensity_range: Range of multiplicative factors to scale stain channel intensities.
        brightness_range: Range for gamma correction to simulate brightness shift.
        always_apply: Whether the transformation should always be applied.
        p: Probability of applying the transformation.
    """

    def __init__(
        self,
        intensity_range: tuple[float, float] = (0.4, 1.5),
        brightness_range: tuple[float, float] = (-0.4, 0.4),
        always_apply: bool = True,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.intensity_range = intensity_range
        self.brightness_range = brightness_range

    def apply(self, img: NDArray, **params) -> NDArray:
        """Apply intensity and brightness adjustments to H and E channels.

        Args:
            img: Image to which the transformation will be applied.
            params: Additional parameters (unused).

        Returns:
            Modified RGB image as a float32 NumPy array with values in [0.0, 1.0].
        """
        hed_image = separate_stains(img, hed_from_rgb)
        h = self.modify_channel(
            hed_image[:, :, 0], self.intensity_range, self.brightness_range
        )
        e = self.modify_channel(
            hed_image[:, :, 1], self.intensity_range, self.brightness_range
        )
        d = hed_image[:, :, 2]

        modified_rgb = combine_stains(np.stack((h, e, d), axis=-1), rgb_from_hed)

        return modified_rgb

    def modify_channel(self, channel: NDArray[np.float32]) -> NDArray[np.float32]:
        intensity_scale = np.random.uniform(*self.intensity_range)
        channel = channel * intensity_scale
        brightness_shift = np.random.uniform(*self.brightness_range)
        channel = exposure.adjust_gamma(channel, gamma=1 + brightness_shift)
        return np.clip(channel, 0, 1)
