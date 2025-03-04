import numpy as np
from albumentations import ImageOnlyTransform
from skimage import exposure


class CombinedModifications(ImageOnlyTransform):
    def __init__(self, intensity_range=(0.4, 1.5), brightness_range=(-0.4, 0.4), always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.intensity_range = intensity_range
        self.brightness_range = brightness_range

    def apply(self, img, **params):
        def modify_channel(channel, intensity_range, brightness_range):
            intensity_scale = np.random.uniform(*intensity_range)
            channel = channel * intensity_scale
            brightness_shift = np.random.uniform(*brightness_range)
            channel = exposure.adjust_gamma(channel, gamma=1 + brightness_shift)
            return np.clip(channel, 0, 1)
        
        h = modify_channel(img[:, :, 0], self.intensity_range, self.brightness_range)
        e = modify_channel(img[:, :, 1], self.intensity_range, self.brightness_range)
        d = img[:, :, 2]  # Skip modification for D channel
        return np.stack((h, e, d), axis=-1)