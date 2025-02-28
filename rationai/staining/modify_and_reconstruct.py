import numpy as np
from skimage.color import combine_stains

from rationai.staining.typing import ModifyFunction, RGBArray, StainTuple
from rationai.staining.utils.inv_mat import inv_mat
from rationai.staining.utils.residual import residual


def modify_and_reconstruct(
    tile: RGBArray,
    modify: ModifyFunction,
    stain0: StainTuple,
    stain1: StainTuple,
    stain2: StainTuple | None = None,
) -> RGBArray:
    """Modifies RGB tile in stain space and converts back to RGB.

    Args:
        tile: Input RGB representation of the region.
        modify: Function that takes three channels in stain space
            and returns their modified versions.
        stain0: First default color vector.
        stain1: Second default color vector.
        stain2: Third default color vector. If not provided,
            a residual vector is computed from the first two.

    Returns:
        Modified region in RGB space.

    Note:
        - To ensure as precise reconstruction as possible,
        **clipping of negative values is ommited** from stain separation.

    References:
        Stain separation is adapted from <a href="https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.separate_stains">skimage.color.separate_stains</a>.

    Example:
    ```python
    import numpy as np
    from skimage.data import immunohistochemistry

    from rationai.staining import modify_and_reconstruct
    from rationai.staining.constants import QUPATH_DAB, QUPATH_H


    def modify(c0, c1, c2):
        # Remove second channel from the image
        c1 = np.zeros_like(c1)

        return c0, c1, c2


    original = immunohistochemistry()
    modified = modify_and_reconstruct(original, modify, QUPATH_H, QUPATH_DAB)
    ```
    """
    if stain2 is None:
        stain2 = residual(stain0, stain1)

    mat = (stain0, stain1, stain2)

    stain2rgb = np.stack(mat)
    rgb2stain = np.stack(inv_mat(mat))

    values = np.maximum(tile.astype(np.float64) / 255, 1e-6)
    log_adjust = np.log(1e-6)

    stains = (np.log(values) / log_adjust) @ rgb2stain
    c0, c1, c2 = tuple(np.moveaxis(stains, -1, 0))

    modified_stains = np.stack(modify(c0, c1, c2), axis=-1)

    return (255 * combine_stains(modified_stains, stain2rgb)).astype(np.uint8)
