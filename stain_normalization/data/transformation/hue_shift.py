

import numpy as np
from albumentations import ImageOnlyTransform
from skimage.color import (
    rgb2hsv, hsv2rgb, 
    rgb2hed, hed2rgb
)


class HueShift(ImageOnlyTransform):
    def __init__(self, hue_shift_range: tuple[float, float] = (0.0, 1.0), always_apply: bool = True, p: float = 1):
        super().__init__(always_apply, p)
        self.hue_shift_range = hue_shift_range

    def apply(self, img, **params):
        hue_shift = np.random.uniform(*self.hue_shift_range)

        h, e, d = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        h_mod = np.clip(h * e, 0, 1)
        e_mod = np.clip(e * 1, 0, 1)
        d_mod = np.clip(d * 1, 0, 1)

        hed_mod = np.stack((h_mod, e_mod, d_mod), axis=-1)

        rgb_image = hed2rgb(hed_mod)
        hsv_image = rgb2hsv(rgb_image)

        # Apply the random hue shift
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 1.0

        modified_rgb = hsv2rgb(hsv_image)
        return rgb2hed(modified_rgb)

