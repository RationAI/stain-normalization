import numpy as np
from PIL.Image import Image

from rationai.staining.typing import StainArray, Tile
from rationai.staining.utils import normalize


def estimate_stain_vectors(
    img: Tile, i0: int = 240, alpha: int = 1, beta: float = 0.15
) -> StainArray:
    """Detects two most dominant staining vectors in the image.

    Args:
        img: Stained image with three RGB channels.
        i0: Transmitted light intensity. Defaults to 240.
        alpha: Percentile offset for robust estimation. Defaults to 1.
        beta: Threshold for transparent pixels in the OD-space. Defaults to 0.15.

    Returns:
        Array with two most dominant color vectors.

    References:
        Adapted from: `https://github.com/schaugf/HEnorm_python`

        Paper: A method for normalizing histology slides for quantitative analysis,
            M Macenko, M Niethammer, JS Marron, D Borland, JT Woosley, G Xiaojun,
            C Schmitt, NE Thomas, IEEE ISBI, 2009. dx.doi.org/10.1109/ISBI.2009.5193250
    """
    if isinstance(img, Image):
        img = np.asarray(img)

    img = img.reshape((-1, 3))

    # Calculate the optical density
    od = -np.log((img.astype(np.float64) + 1) / i0)  # (H*W, 3)

    # remove transparent pixels
    od_hat = od[~np.any(od < beta, axis=-1)]

    if od_hat.size <= 6:
        # Not enough stained pixels remained, return empty vectors
        return np.full(shape=(2, 3), fill_value=np.nan)

    # Compute eigenvectors
    _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))

    # Project on the plane spanned by the eigenvectors
    # corresponding to the two largest eigenvalues.
    t_hat = od_hat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])

    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    v_min = eigvecs[:, 1:3].dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
    v_max = eigvecs[:, 1:3].dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)

    # Make hematoxylin first and eosin second (heuristic)
    if v_min[0] > v_max[0]:
        return np.array((normalize(v_min[:, 0]), normalize(v_max[:, 0])))

    return np.array((normalize(v_max[:, 0]), normalize(v_min[:, 0])))
