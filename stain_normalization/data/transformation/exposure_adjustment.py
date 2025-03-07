import numpy as np
from albumentations import ImageOnlyTransform
from skimage.color import separate_stains, combine_stains, hed_from_rgb, rgb_from_hed


class ExposureAdjustment(ImageOnlyTransform):
    def __init__(self, brightness_range: tuple[float, float] = (0.8, 1.2), always_apply: bool = True, p: float = 1):
        super().__init__(always_apply, p)
        self.brightness_range = brightness_range

    def apply(self, img, **params):
        hed_img = separate_stains(img, hed_from_rgb)

        brightness_factor = np.random.uniform(*self.brightness_range)
        h, e, d = hed_img[:, :, 0], hed_img[:, :, 1], hed_img[:, :, 2]
        h = np.clip(h * brightness_factor, 0, 1)
        e = np.clip(e * brightness_factor, 0, 1)
        d = np.clip(d * brightness_factor, 0, 1)
        
        modified_rgb = combine_stains(np.stack((h, e, d), axis=-1), rgb_from_hed)
        return modified_rgb
