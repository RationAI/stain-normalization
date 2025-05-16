from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from rationai.staining import estimate_stain_vectors
from skimage.metrics import structural_similarity as ssim


class AnalysisExport(Callback):
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
        return estimate_stain_vectors(img, i0=240, alpha=1, beta=0.15)

    def _compare_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.linalg.norm(vec1 - vec2))

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        return float(ssim(img1, img2, channel_axis=-1, data_range=255))

    def _compute_nmi(self, img: np.ndarray): # normalized median intensity
        avg_rgb = img.mean(axis=2)
        median_val = np.median(avg_rgb)
        p95_val = np.percentile(avg_rgb, 95)
        nmi = median_val / p95_val if p95_val != 0 else 0
        return nmi

def on_test_batch_end(
    self,
    trainer: Trainer,
    pl_module: LightningModule,
    outputs: list[torch.Tensor],
    batch: tuple[torch.Tensor, list],
    batch_idx: int,
    dataloader_idx: int = 0,
) -> None:
    for b in range(len(outputs)):
        predicted_tensor = outputs[b]
        original_np = batch[1][b]["original_image"].astype("uint8")
        modified_np = (batch[1][b]["modified_image"] * 255).astype("uint8")
        index = batch[1][b]["index"]
        predicted_np = self.tensor_to_image(predicted_tensor)

        vec_original = self._estimate_vectors(original_np)
        vec_modified = self._estimate_vectors(modified_np)
        vec_predicted = self._estimate_vectors(predicted_np)

        vec_diff_mod = self._compare_vectors(vec_original, vec_modified)
        vec_diff_pred = self._compare_vectors(vec_original, vec_predicted)

        ssim_mod = self._compute_ssim(original_np, modified_np)
        ssim_pred = self._compute_ssim(original_np, predicted_np)

        
        nmi_original = self.compute_nmi(original_np)
        nmi_modified = self.compute_nmi(modified_np)
        nmi_predicted = self.compute_nmi(predicted_np)

        diff_row = {
            "index": index,
            "vector_diff_modified_vs_original": vec_diff_mod,
            "vector_diff_predicted_vs_original": vec_diff_pred,
            "ssim_modified_vs_original": ssim_mod,
            "ssim_predicted_vs_original": ssim_pred,
            "nmi_diff_modified_vs_original": nmi_modified - nmi_original,
            "nmi_diff_predicted_vs_original": nmi_predicted - nmi_original,
        }

        raw_row = {"index": index}
        for name, vec, nmi in zip(
            ["original", "modified", "predicted"],
            [vec_original, vec_modified, vec_predicted],
            [nmi_original, nmi_modified, nmi_predicted], strict=False
        ):
            vectors = vec.flatten()
            for j, val in enumerate(vectors):
                raw_row[f"{name}_vec_{j}"] = val
            raw_row[f"{name}_nmi"] = nmi

        self.df_diff = pd.concat([self.df_diff, pd.DataFrame([diff_row])], ignore_index=True)
        self.df_raw = pd.concat([self.df_raw, pd.DataFrame([raw_row])], ignore_index=True)

    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list,
    ) -> None:
        diff_path = self.output_dir / "analysis_differences.csv"
        raw_path = self.output_dir / "analysis_raw_vectors.csv"

        self.df_diff.to_csv(diff_path, index=False)
        self.df_raw.to_csv(raw_path, index=False)

        mlflow.log_artifact(str(diff_path), artifact_path="analysis_metrics")
        mlflow.log_artifact(str(raw_path), artifact_path="analysis_metrics")
