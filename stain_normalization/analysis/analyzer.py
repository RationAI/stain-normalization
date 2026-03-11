"""Stain normalization analysis tool.

Compares images using selected metrics, accumulates results,
and provides statistics and saving.
"""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from rationai.staining import estimate_stain_vectors
from skimage.metrics import structural_similarity

from stain_normalization.metrics.image_metrics import (
    compute_lab_brightness_psnr,
    compute_nmi,
    compute_pcc,
)
from stain_normalization.metrics.vector_metrics import compare_vectors


class StainAnalyzer:
    """Compare images using selected metrics, accumulate results.

    Args:
        reference: Optional fixed reference image. If given, stain vectors
            and NMI are pre-computed once.
        metrics: Which metrics to compute. None = all.
    """

    AVAILABLE_METRICS: ClassVar[list[str]] = [
        "vectors",
        "ssim",
        "pcc",
        "nmi",
        "lab_psnr",
    ]
    PAIRED_ONLY: ClassVar[set[str]] = {"ssim", "pcc", "lab_psnr"}

    def __init__(
        self,
        reference: np.ndarray[Any, Any] | None = None,
        metrics: list[str] | None = None,
    ) -> None:
        self.metrics = metrics or self.AVAILABLE_METRICS
        self._results: list[dict[str, Any]] = []

        for m in self.metrics:
            if m not in self.AVAILABLE_METRICS:
                raise ValueError(
                    f"Unknown metric '{m}'. Available: {self.AVAILABLE_METRICS}"
                )

        # if we have reference image, precompute stain vectors and NMI
        self._ref_img = reference
        self._ref_vectors = None
        self._ref_nmi = None

        if reference is not None:
            if "vectors" in self.metrics:
                self._ref_vectors = estimate_stain_vectors(reference)
            if "nmi" in self.metrics:
                self._ref_nmi = compute_nmi(reference)

    @property
    def results(self) -> pd.DataFrame:
        """Accumulated comparison results as DataFrame."""
        return pd.DataFrame(self._results)

    def clear(self) -> None:
        """Clear accumulated results."""
        self._results.clear()

    def compare(
        self,
        image: np.ndarray[Any, Any],
        image_id: str | None = None,
        reference: np.ndarray[Any, Any] | None = None,
    ) -> dict[str, Any]:
        """Compare an image against the reference and store the result.

        Args:
            image: Image to compare.
            image_id: Optional identifier for this comparison.
            reference: Override reference image for this call.

        Returns:
            Dict with metric results.
        """
        ref_img = reference if reference is not None else self._ref_img
        if ref_img is None:
            raise ValueError(
                "No reference image. Pass one to __init__ or to compare()."
            )

        is_paired = reference is not None

        if reference is not None:
            ref_vectors = (
                estimate_stain_vectors(ref_img) if "vectors" in self.metrics else None
            )
            ref_nmi = compute_nmi(ref_img) if "nmi" in self.metrics else None
        else:
            ref_vectors = self._ref_vectors
            ref_nmi = self._ref_nmi

        result: dict[str, Any] = {"id": image_id} if image_id is not None else {}

        if "vectors" in self.metrics:
            assert (
                ref_vectors is not None
            )  # refrence image deosnt have stain vectors (too much background)
            img_vectors = estimate_stain_vectors(image)
            vec_result = compare_vectors(ref_vectors, img_vectors)
            result.update(vec_result)
            img_vectors_paired = (
                img_vectors[[1, 0]] if vec_result["was_swapped"] else img_vectors
            )
            for j, val in enumerate(ref_vectors.flatten()):
                result[f"ref_vec_{j}"] = float(val)
            for j, val in enumerate(img_vectors_paired.flatten()):
                result[f"img_vec_{j}"] = float(val)

        if "ssim" in self.metrics and is_paired:
            result["ssim"] = float(
                structural_similarity(
                    ref_img,
                    image,
                    channel_axis=-1,
                    data_range=255,
                )
            )

        if "pcc" in self.metrics and is_paired:
            result["pcc"] = compute_pcc(ref_img, image)

        if "nmi" in self.metrics:
            assert ref_nmi is not None
            img_nmi = compute_nmi(image)
            result["ref_nmi"] = ref_nmi
            result["nmi"] = img_nmi
            result["nmi_diff"] = img_nmi - ref_nmi

        if "lab_psnr" in self.metrics and is_paired:
            result["lab_brightness_psnr"] = compute_lab_brightness_psnr(ref_img, image)

        self._results.append(result)
        return result

    # --- Statistics ---

    def get_statistics(self) -> pd.DataFrame:
        """Summary statistics for accumulated results.

        Returns:
            DataFrame with mean, std, min, max, percentiles.
        """
        df = self.results
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

    def get_baseline_ranges(
        self,
        percentile_low: float = 5,
        percentile_high: float = 95,
    ) -> dict[str, tuple[float, float]]:
        """Get acceptable value ranges for each metric.

        Args:
            percentile_low: Lower percentile bound. Defaults to 5.
            percentile_high: Upper percentile bound. Defaults to 95.

        Returns:
            Dict mapping metric names to (low, high) tuples.
        """
        df = self.results
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return {
            col: (
                float(df[col].quantile(percentile_low / 100)),
                float(df[col].quantile(percentile_high / 100)),
            )
            for col in numeric_cols
        }

    def save_csv(self, output_dir: str | Path) -> Path:
        """Save accumulated results and statistics as CSV files.

        Args:
            output_dir: Directory to save files into.

        Returns:
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.results.to_csv(output_dir / "results.csv", index=True)
        self.get_statistics().to_csv(output_dir / "statistics.csv")

        ranges = self.get_baseline_ranges()
        with open(output_dir / "baseline_ranges.txt", "w") as f:
            f.write("Baseline Metric Ranges (5th-95th percentile)\n")
            f.write("=" * 55 + "\n\n")
            for metric, (low, high) in sorted(ranges.items()):
                f.write(f"{metric:30s}: [{low:8.4f}, {high:8.4f}]\n")

        return output_dir
