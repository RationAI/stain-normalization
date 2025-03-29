from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from stain_normalization.modeling import L1SSIMLoss
from stain_normalization.typing import Input, Outputs
from torch.optim import Adam
from torch.optim.optimizer import Optimizer 
from torchmetrics.image import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
from torchmetrics import  MetricCollection
from typing import Any

class StainNormalizationModel(LightningModule):
    def __init__(self, backbone: nn.Module, decode_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.criterion = L1SSIMLoss() 

        self.val_metrics = MetricCollection(
            {
                "ssim": StructuralSimilarityIndexMeasure(),
                "psnr": PeakSignalNoiseRatio()
            }
        )  
        self.test_metrics = self.val_metrics.clone(prefix="test/")
        self.val_metrics.prefix = "validation/"

    def forward(self, x: Input) -> Outputs:
        features = self.backbone(x)
        return self.decode_head(features)

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
        self.log("validation/loss", loss, on_step=True, on_epoch=False)
        self.val_metrics.update(outputs, targets)
        self.log_dict(
            self.val_metrics,
            batch_size=len(inputs),
            on_epoch=False, 
            on_step=True
        )
        

    def test_step(self, batch: Input) -> None:
        inputs, targets = batch
        outputs = self(inputs)
        self.test_metrics.update(outputs, targets)
        self.log_dict(
            self.test_metrics,
            batch_size=len(inputs),
            on_epoch=True,
        )    

    def predict_step(self, batch: tuple[Tensor, Any], batch_idx: int, dataloader_idx: int = 0) -> Outputs:
        inputs = batch[0]
        return self(inputs)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=1e-4)
