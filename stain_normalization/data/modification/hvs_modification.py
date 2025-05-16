

import numpy as np
from albumentations import ImageOnlyTransform
from skimage.color import hsv2rgb, rgb2hsv


class HVSModification(ImageOnlyTransform):
    def __init__(
        self,
        hue_shift_range: tuple[float, float] = (-0.2, 0.2),
        saturation_range: tuple[float, float] = (0.8, 1.5),
        value_range: tuple[float, float] = (0.8, 1.3),
        always_apply: bool = True,
        p: float = 1.0
    ):

        super().__init__(always_apply, p)
        self.hue_shift_range = hue_shift_range
        self.saturation_range = saturation_range
        self.value_range = value_range

    def apply(self, img, **params):
        hue_shift = np.random.uniform(*self.hue_shift_range)
        saturation_scale = np.random.uniform(*self.saturation_range)
        value_scale = np.random.uniform(*self.value_range)

        hsv_image = rgb2hsv(img)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 1.0
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 1)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * value_scale, 0, 1)

        modified_rgb = hsv2rgb(hsv_image)

        return modified_rgb