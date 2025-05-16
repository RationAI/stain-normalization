import numpy as np
from albumentations import ImageOnlyTransform
from skimage.color import combine_stains, hed_from_rgb, rgb_from_hed, separate_stains


class HEDFactor(ImageOnlyTransform):
    def __init__(self, 
                 h_range: tuple[float, float] = (0.8, 1.2), 
                 e_range: tuple[float, float] = (0.8, 1.2), 
                 always_apply: bool = True, 
                 p: float = 1.0):
        super().__init__(always_apply, p)
        self.h_range = h_range
        self.e_range = e_range

    def apply(self, img, **params):        
        h_factor = np.random.uniform(*self.h_range)
        e_factor = np.random.uniform(*self.e_range)

        return self.multiply_channels(img, e_factor, h_factor)

    def multiply_channels(self, image, e_factor=1.1, h_factor=1.1,):
        hed_image = separate_stains(image, hed_from_rgb)
        h = hed_image[:, :, 0] * h_factor
        e = hed_image[:, :, 1] * e_factor
        d = hed_image[:, :, 2] 
        modified_rgb = combine_stains(np.stack((h, e, d), axis=-1), rgb_from_hed)
        return modified_rgb
