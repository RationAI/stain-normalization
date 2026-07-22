from typing import Any

import numpy as np
from albumentations import Transform3D
from numpy.typing import NDArray

from stain_normalization.data.artifact.fractal_mask import FractalMask


class Artifact:
    """Add a synthetic artifact to an (input, target) pair.

    Unlike modify, which only degrades the input, this adds a non-H&E color
    (extreme_modification) to a mask region of the tissue identically into BOTH
    input and target, simulating dyes and other artifacts, so the model learns to
    keep such non-H&E colors unchanged.

    Attributes:
        extreme_modification: Transform producing the extreme dye color, applied to the tile.
        mask: Generator producing the artifact region shape.
        bg_threshold: Pixels brighter than this in every channel count as background and are left dye-free (the HSV recolor cannot color white).
        p: Probability that an artifact is applied to a tile.
    """

    def __init__(
        self,
        extreme_modification: Transform3D,
        mask: FractalMask,
        bg_threshold: float = 0.85,
        p: float = 1.0,
    ) -> None:
        self.extreme_modification = extreme_modification
        self.mask = mask
        self.bg_threshold = bg_threshold
        self.p = p

    def apply(
        self,
        tile: NDArray[np.uint8],
        model_input: NDArray[Any],
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.bool_]]:
        """Add the artifact to the modified input and the target.

        Args:
            tile: Original uint8 RGB tile, values in [0, 255].
            model_input: Modified input as a float array in [0, 1].

        Returns:
            An (input, target, mask) triple in [0, 1]; input and target are
            unchanged and the mask all-False when the mask does not fire.
        """
        target = tile / 255.0
        if np.random.uniform() >= self.p:
            return model_input, target, np.zeros(tile.shape[:2], dtype=bool)

        mask = self.get_mask(tile)
        if not mask.any():
            return model_input, target, mask

        extreme = self.extreme_modification(image=tile)["image"]
        selector = mask[..., np.newaxis]
        return (
            np.where(selector, extreme, model_input),
            np.where(selector, extreme, target),
            mask,
        )

    def get_mask(self, tile: NDArray[np.uint8]) -> NDArray[np.bool_]:
        """The artifact region: the fractal mask restricted to tissue.

        Args:
            tile: Original uint8 RGB tile, values in [0, 255].

        Returns:
            Boolean mask, True where the artifact is added (mask region that
            overlaps tissue, i.e. not white background); all-False when the mask
            does not fire.
        """
        mask = self.mask.generate(*tile.shape[:2])
        if mask.any():
            mask &= tile.min(axis=2) / 255.0 < self.bg_threshold
        return mask
