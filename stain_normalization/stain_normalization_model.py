from typing import Any

from lightning import LightningModule
from torch import Tensor, stack
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError

from stain_normalization.modeling import L1SSIMLoss, UNet
from stain_normalization.typing import Input, Outputs


class StainNormalizationModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet(in_channels=3, out_channels=3)
        self.criterion = L1SSIMLoss()

        self.val_metrics = MetricCollection(
            {
                "ssim": StructuralSimilarityIndexMeasure(),
                "l1": MeanAbsoluteError()

            }
        )
        self.test_metrics = self.val_metrics.clone(prefix="test/")
        self.val_metrics.prefix = "validation/"

    def forward(self, x: Input) -> Outputs:
        return self.unet(x)

    def training_step(self, batch: Input) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Input) -> None:
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

    def test_step(self, batch: Input) -> Outputs:
        inputs, data = batch
        outputs = self(inputs)
        targets = stack([item["original_image_tensor"] for item in data])
        self.test_metrics.update(outputs, targets)
        self.log_dict(
            self.test_metrics,
            batch_size=len(inputs),
            on_epoch=True,
        )
        return outputs

    def predict_step(self, batch: tuple[Tensor, Any], batch_idx: int) -> Outputs:
        inputs = batch[0]
        return self(inputs)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=1e-4)
