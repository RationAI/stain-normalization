from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from rationai.staining import estimate_stain_vectors
from skimage.metrics import structural_similarity as ssim


class AnalysisExport(Callback):
    """Callback for exporting analysis metrics during testing."""

    def __init__(self, output_dir: str | Path, normalization_config) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        normalization = normalization_config
        self.mean = torch.tensor(normalization.mean).view(3, 1, 1)
        self.std = torch.tensor(normalization.std).view(3, 1, 1)

        self.df_diff = pd.DataFrame()
        self.df_raw = pd.DataFrame()

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        std = self.std.to(tensor.device)
        mean = self.mean.to(tensor.device)
        return (tensor * std) + mean

    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = self.denormalize(tensor).clamp(0, 1)
        return (tensor * 255).byte().permute(1, 2, 0).cpu().numpy()

    def _estimate_vectors(self, img: np.ndarray) -> np.ndarray:
        """Estimate stain vectors for a given image."""
        return estimate_stain_vectors(img, i0=240, alpha=1, beta=0.15)

    def _compare_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute Euclidean distance between two stain vector estimates."""
        return float(np.linalg.norm(vec1 - vec2))

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Structural Similarity Index (SSIM) between two images."""
        return float(ssim(img1, img2, channel_axis=-1, data_range=255))

    def _compute_nmi(self, img: np.ndarray) -> float:
        """Compute Normalized Median Intensity (NMI) of an image."""
        avg_rgb = img.mean(axis=2)
        median_val = np.median(avg_rgb)
        p95_val = np.percentile(avg_rgb, 95)
        return median_val / p95_val if p95_val != 0 else 0.0

    def _compute_pcc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Pearson Correlation Coefficient (PCC) between two images."""
        img1_flat = img1.flatten().astype(np.float64)
        img2_flat = img2.flatten().astype(np.float64)
        if img1_flat.size == 0 or img2_flat.size == 0:
            return 0.0

        return float(np.corrcoef(img1_flat, img2_flat)[0, 1])

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list[torch.Tensor],
        batch: tuple[torch.Tensor, list[dict]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called at the end of each test batch.

        Computes metrics for each sample and accumulates results.
        """
        for b in range(len(outputs)):
            predicted_tensor = outputs[b]
            original_img = batch[1][b]["original_image"].astype("uint8")
            modified_img = (batch[1][b]["modified_image"] * 255).astype("uint8")
            index = batch[1][b]["index"]
            predicted_img = self.tensor_to_image(predicted_tensor)

            vec_original = self._estimate_vectors(original_img)
            vec_modified = self._estimate_vectors(modified_img)
            vec_predicted = self._estimate_vectors(predicted_img)

            vec_diff_mod = self._compare_vectors(vec_original, vec_modified)
            vec_diff_pred = self._compare_vectors(vec_original, vec_predicted)

            ssim_mod = self._compute_ssim(original_img, modified_img)
            ssim_pred = self._compute_ssim(original_img, predicted_img)

            nmi_original = self._compute_nmi(original_img)
            nmi_modified = self._compute_nmi(modified_img)
            nmi_predicted = self._compute_nmi(predicted_img)

            pcc_mod = self._compute_pcc(original_img, modified_img)
            pcc_pred = self._compute_pcc(original_img, predicted_img)

            diff_row = {
                "index": index,
                "vector_diff_modified_vs_original": vec_diff_mod,
                "vector_diff_predicted_vs_original": vec_diff_pred,
                "ssim_modified_vs_original": ssim_mod,
                "ssim_predicted_vs_original": ssim_pred,
                "nmi_diff_modified_vs_original": nmi_modified - nmi_original,
                "nmi_diff_predicted_vs_original": nmi_predicted - nmi_original,
                "pcc_modified_vs_original": pcc_mod,
                "pcc_predicted_vs_original": pcc_pred,
            }

            raw_row = {"index": index}
            for name, vec, nmi in zip(
                ["original", "modified", "predicted"],
                [vec_original, vec_modified, vec_predicted],
                [nmi_original, nmi_modified, nmi_predicted],
                strict=False,
            ):
                vectors = vec.flatten()
                for j, val in enumerate(vectors):
                    raw_row[f"{name}_vec_{j}"] = val
                raw_row[f"{name}_nmi"] = nmi

            self.df_diff = pd.concat(
                [self.df_diff, pd.DataFrame([diff_row])], ignore_index=True
            )
            self.df_raw = pd.concat(
                [self.df_raw, pd.DataFrame([raw_row])], ignore_index=True
            )

    def on_test_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list,
    ) -> None:
        """Called at the end of testing.

        Saves the collected metrics as CSV files and logs them as mlflow artifacts.
        """
        diff_path = self.output_dir / "analysis_differences.csv"
        raw_path = self.output_dir / "analysis_raw_vectors.csv"

        self.df_diff.to_csv(diff_path, index=False)
        self.df_raw.to_csv(raw_path, index=False)

        mlflow.log_artifact(str(diff_path), artifact_path="analysis_metrics")
        mlflow.log_artifact(str(raw_path), artifact_path="analysis_metrics")
