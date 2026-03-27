from pathlib import Path
from typing import Any

import mlflow
import torch
from lightning import LightningModule, Trainer

from stain_normalization.analysis.analyzer import StainAnalyzer
from stain_normalization.callbacks._base import ImageCallback
from stain_normalization.type_aliases import Outputs


class AnalysisExport(ImageCallback):
    """Exports analysis metrics during testing."""

    def __init__(self, output_dir: str | Path) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mod_analyzer = StainAnalyzer()
        self.pred_analyzer = StainAnalyzer()

    def on_test_batch_end(  # type: ignore[override]  # narrowed Lightning STEP_OUTPUT
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: tuple[torch.Tensor, list[dict[str, Any]]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Computes metrics for each sample and accumulates results."""
        for b in range(len(outputs)):
            original_img = batch[1][b]["original_image"].astype("uint8")
            modified_img = (batch[1][b]["modified_image"] * 255).astype("uint8")
            predicted_img = self.tensor_to_image(outputs[b])

            self.mod_analyzer.compare(modified_img, paired_image=original_img)
            self.pred_analyzer.compare(predicted_img, paired_image=original_img)

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
