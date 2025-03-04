import numpy as np
from albumentations import ImageOnlyTransform


class ExposureAdjustment(ImageOnlyTransform):
    def __init__(self, brightness_range: tuple[float, float] = (0.8, 1.2), always_apply: bool = True, p: float = 1):
        super().__init__(always_apply, p)
        self.brightness_range = brightness_range

    def apply(self, img, **params):
        brightness_factor = np.random.uniform(*self.brightness_range)
        h, e, d = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        h = np.clip(h * brightness_factor, 0, 1)
        e = np.clip(e * brightness_factor, 0, 1)
        d = np.clip(d * brightness_factor, 0, 1)
        return np.stack((h, e, d), axis=-1)

