import numpy as np
from albumentations import ImageOnlyTransform
from skimage import exposure
from skimage.color import combine_stains, hed_from_rgb, rgb_from_hed, separate_stains


class CombinedModifications(ImageOnlyTransform):
    def __init__(self, intensity_range=(0.4, 1.5), brightness_range=(-0.4, 0.4), always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.intensity_range = intensity_range
        self.brightness_range = brightness_range

    def apply(self, hed_img, **params):
        def modify_channel(channel, intensity_range, brightness_range):
            intensity_scale = np.random.uniform(*intensity_range)
            channel = channel * intensity_scale
            brightness_shift = np.random.uniform(*brightness_range)
            channel = exposure.adjust_gamma(channel, gamma=1 + brightness_shift)
            return np.clip(channel, 0, 1)

        hed_image = separate_stains(hed_img, hed_from_rgb)
        h = modify_channel(hed_image[:, :, 0], self.intensity_range, self.brightness_range)
        e = modify_channel(hed_image[:, :, 1], self.intensity_range, self.brightness_range)
        d = hed_image[:, :, 2]  # Skip modification for D channel

        modified_rgb = combine_stains(np.stack((h, e, d), axis=-1), rgb_from_hed)
        return modified_rgb