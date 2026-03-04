from pathlib import Path

import mlflow
import torch
from lightning import LightningModule, Trainer
from omegaconf import DictConfig

from ..analysis.analyzer import StainAnalyzer
from ._base import NormalizationCallback


class AnalysisExport(NormalizationCallback):
    """Exports analysis metrics during testing."""

    def __init__(self, output_dir: str | Path, normalization_config: DictConfig) -> None:
        super().__init__(normalization_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mod_analyzer = StainAnalyzer()
        self.pred_analyzer = StainAnalyzer()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list[torch.Tensor],
        batch: tuple[torch.Tensor, list[dict]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Computes metrics for each sample and accumulates results."""
        for b in range(len(outputs)):
            original_img = batch[1][b]["original_image"].astype("uint8")
            modified_img = (batch[1][b]["modified_image"] * 255).astype("uint8")
            predicted_img = self.tensor_to_image(outputs[b])

            self.mod_analyzer.compare(modified_img, reference=original_img)
            self.pred_analyzer.compare(predicted_img, reference=original_img)

    def on_test_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Saves collected metrics as CSV files and logs them as mlflow artifacts."""
        metrics_dir = self.output_dir / "analysis_metrics"

        mod_dir = self.mod_analyzer.save_csv(metrics_dir / "modified")
        pred_dir = self.pred_analyzer.save_csv(metrics_dir / "predicted")

        for f in mod_dir.glob("*"):
            mlflow.log_artifact(str(f), artifact_path="analysis_metrics/modified")
        for f in pred_dir.glob("*"):
            mlflow.log_artifact(str(f), artifact_path="analysis_metrics/predicted")
