import numpy as np
from skimage.color import rgb2lab


def _od_to_lab(od_vector: np.ndarray) -> np.ndarray:
    """Convert optical density vector to Lab color.

    Args:
        od_vector: Stain vector in optical density space.

    Returns:
        Color in Lab space as [L, a, b].
    """
    # Calculate RGB from optical density by reversing the process in estimate_stain_vectors.
    # default i0=240 (transmitted light intensity)
    rgb = np.clip(240 * np.exp(-od_vector), 0, 255) / 255.0
    return rgb2lab(rgb.reshape(1, 1, 3)).flatten()


def delta_e76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIE76 Delta E with dL=0 (chromaticity only).

    CIE76 Delta E is sqrt(dL^2 + da^2 + db^2). We set dL=0 because we compare
    dyes not colors, so brightness is irrelevant.

    Args:
        lab1: First color in Lab space.
        lab2: Second color in Lab space.

    Returns:
        sqrt(da^2 + db^2).
    """
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return float(np.sqrt(da**2 + db**2))


def compare_vectors(
    vecs1: np.ndarray,
    vecs2: np.ndarray,
) -> dict:
    """Compare two sets of stain vectors in Lab chromaticity space.

    Args:
        vecs1: Stain vectors from the first image in OD space.
        vecs2: Stain vectors from the second image in OD space.

    Returns:
        Dict with d_hematoxylin, d_eosin and was_swapped.
        Returns NaN distances if either vector set contains NaN.
    """
    if np.any(np.isnan(vecs1)) or np.any(np.isnan(vecs2)):
        return {
            'd_hematoxylin': float('nan'), 'd_eosin': float('nan'),
            'was_swapped': False,
        }


    sim_straight = np.dot(vecs1[0], vecs2[0]) + np.dot(vecs1[1], vecs2[1])
    sim_swapped = np.dot(vecs1[0], vecs2[1]) + np.dot(vecs1[1], vecs2[0])
    was_swapped = sim_swapped > sim_straight
    vecs2_paired = vecs2[[1, 0]] if was_swapped else vecs2

    lab1_a = _od_to_lab(vecs1[0])
    lab1_b = _od_to_lab(vecs1[1])
    lab2_a = _od_to_lab(vecs2_paired[0])
    lab2_b = _od_to_lab(vecs2_paired[1])

    return {
        'd_hematoxylin': delta_e76(lab1_a, lab2_a),
        'd_eosin': delta_e76(lab1_b, lab2_b),
        'was_swapped': was_swapped,
    }


