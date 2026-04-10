from lightning import LightningModule
from torch import Tensor, stack
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError

import torch

from stain_normalization.metrics.torch_metrics import (
    MeanBrightness,
    MeanLabPSNR,
    MeanPCC,
    MeanStainDistance,
)
from stain_normalization.modeling import L1SSIMLoss, UNet
from stain_normalization.type_aliases import Batch, Outputs, PredictBatch


class StainNormalizationModel(LightningModule):
    def __init__(
        self,
        normalize_mean: list[float],
        normalize_std: list[float],
        lr: float = 1e-4,
        lambda_dssim: float = 0.6,
        lambda_l1: float = 0.2,
        lambda_lum: float = 0.2,
        lambda_gdl: float = 0.1,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.unet = UNet(in_channels=3, out_channels=3)
        self.criterion = L1SSIMLoss(
            lambda_dssim=lambda_dssim,
            lambda_l1=lambda_l1,
            lambda_lum=lambda_lum,
            lambda_gdl=lambda_gdl,
        )

        self.register_buffer(
            "_denorm_mean", torch.tensor(normalize_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_denorm_std", torch.tensor(normalize_std).view(1, 3, 1, 1)
        )

        val_metrics = MetricCollection(
            {"ssim": StructuralSimilarityIndexMeasure(), "l1": MeanAbsoluteError()}
        )
        self.val_metrics = val_metrics.clone(prefix="validation/")

        self.test_metrics = MetricCollection(
            {
                "ssim": StructuralSimilarityIndexMeasure(),
                "l1": MeanAbsoluteError(),
                "pcc": MeanPCC(),
                "brightness": MeanBrightness(),
                "lab_psnr": MeanLabPSNR(),
                "d_hematoxylin": MeanStainDistance("d_hematoxylin"),
                "d_eosin": MeanStainDistance("d_eosin"),
            },
            prefix="test/",
            compute_groups=False,
        )

    def forward(self, x: Tensor) -> Outputs:
        return self.unet(x)

    def _denormalize(self, tensor: Tensor) -> Tensor:
        std: Tensor = self._denorm_std  # type: ignore[assignment]
        mean: Tensor = self._denorm_mean  # type: ignore[assignment]
        return (tensor * std + mean).clamp(0, 1)

    def training_step(self, batch: Batch) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Batch) -> None:
        inputs, targets = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log("validation/loss", loss, on_step=False, on_epoch=True, logger=True)
        self.val_metrics.update(outputs, targets)
        self.log_dict(
            self.val_metrics,
            batch_size=len(inputs),
            on_epoch=True,
        )

    def test_step(self, batch: PredictBatch, batch_idx: int = 0) -> Outputs:
        inputs, data = batch
        outputs = self(inputs)
        targets = stack([item["original_image_tensor"] for item in data]).to(
            outputs.device
        )
        denormed_outputs = self._denormalize(outputs)
        denormed_targets = self._denormalize(targets)
        self.test_metrics.update(denormed_outputs, denormed_targets)
        self.log_dict(
            self.test_metrics,
            batch_size=len(inputs),
            on_epoch=True,
        )
        return denormed_outputs

    def predict_step(
        self, batch: PredictBatch, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Outputs:
        inputs = batch[0]
        return self._denormalize(self(inputs))

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.lr)
