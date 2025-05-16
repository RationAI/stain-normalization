import numpy as np
from albumentations import ImageOnlyTransform


class ExposureAdjustment(ImageOnlyTransform):
    def __init__(self, brightness_range: tuple[float, float] = (0.8, 1.2), always_apply: bool = True, p: float = 1):
        super().__init__(always_apply, p)
        self.brightness_range = brightness_range

    def apply(self, img, **params):
        brightness_factor = np.random.uniform(*self.brightness_range)
        img = img.astype(np.float32)
        img = np.clip(img*brightness_factor, 0.0, 1.0)
        
        return img
