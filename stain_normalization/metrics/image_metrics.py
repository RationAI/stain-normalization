from typing import Any

import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import peak_signal_noise_ratio


def compute_nmi(img: np.ndarray[Any, Any]) -> float:
    """Normalized Median Intensity — measures relative brightness of an image.

    Args:
        img: RGB image.

    Returns:
        Ratio of median to 95th percentile intensity.
    """
    avg_rgb = img.mean(axis=2)
    median_val = np.median(avg_rgb)
    p95_val = np.percentile(avg_rgb, 95)

    if p95_val == 0:
        return 0.0

    return float(median_val / p95_val)


def compute_pcc(img1: np.ndarray[Any, Any], img2: np.ndarray[Any, Any]) -> float:
    """Pearson Correlation Coefficient between two images.

    Args:
        img1: First image.
        img2: Second image.

    Returns:
        PCC value, or 0.0 if either image has zero variance.
    """
    img1_flat = img1.flatten().astype(np.float64)
    img2_flat = img2.flatten().astype(np.float64)

    if np.std(img1_flat) == 0 or np.std(img2_flat) == 0:
        return 0.0

    return float(np.corrcoef(img1_flat, img2_flat)[0, 1])


def compute_mean_brightness(img: np.ndarray[Any, Any]) -> float:
    """Mean L* brightness of an image in CIE Lab color space.

    Args:
        img: RGB image (uint8).

    Returns:
        Mean L* value (0–100 scale, higher = brighter).
    """
    lab = rgb2lab(img)
    return float(lab[:, :, 0].mean())


def compute_lab_brightness_psnr(
    img1: np.ndarray[Any, Any], img2: np.ndarray[Any, Any]
) -> float:
    """PSNR on the L* channel in Lab color space.

    Args:
        img1: First RGB image.
        img2: Second RGB image.

    Returns:
        PSNR in dB on the lightness channel.
    """
    lab1 = rgb2lab(img1.astype(np.float32) / 255.0)
    lab2 = rgb2lab(img2.astype(np.float32) / 255.0)
    return float(
        peak_signal_noise_ratio(lab1[:, :, 0], lab2[:, :, 0], data_range=100.0)
    )
