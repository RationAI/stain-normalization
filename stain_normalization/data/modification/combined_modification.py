from typing import Any

import numpy as np
from albumentations import ImageOnlyTransform
from numpy.typing import NDArray
from skimage import exposure
from skimage.color import combine_stains, hed_from_rgb, rgb_from_hed, separate_stains


class CombinedModifications(ImageOnlyTransform):  # type: ignore[misc]  # untyped import
    """Apply combined modifications to the H&E channels in HED color space.

    Attributes:
        od_scale_range: Range of multiplicative factors to scale stain OD (optical density) values.
        brightness_range: Range for gamma correction to simulate brightness shift.
        p: Probability of applying the transformation.
    """

    def __init__(
        self,
        od_scale_range: tuple[float, float] = (0.4, 1.5),
        brightness_range: tuple[float, float] = (-0.4, 0.4),
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.od_scale_range = od_scale_range
        self.brightness_range = brightness_range

    def apply(self, img: NDArray[Any], **params: Any) -> NDArray[Any]:
        """Apply OD scaling and brightness adjustments to H and E channels.

        Args:
            img: Image to which the transformation will be applied.
            params: Additional parameters (unused).

        Returns:
            Modified RGB image as a float32 NumPy array with values in [0.0, 1.0].
        """
        hed_image = separate_stains(img, hed_from_rgb)
        h = self.modify_channel(hed_image[:, :, 0])
        e = self.modify_channel(hed_image[:, :, 1])
        d = hed_image[:, :, 2]

        modified_rgb = combine_stains(np.stack((h, e, d), axis=-1), rgb_from_hed)

        return np.clip(modified_rgb, 0, 1).astype(np.float32)

    def modify_channel(self, channel: NDArray[np.float32]) -> NDArray[np.float32]:
        od_scale = np.random.uniform(*self.od_scale_range)
        channel = channel * od_scale
        brightness_shift = np.random.uniform(*self.brightness_range)
        return exposure.adjust_gamma(channel, gamma=1 + brightness_shift)  # type: ignore[return-value]
