from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer
from omegaconf import DictConfig
from PIL import Image

from stain_normalization.callbacks._base import NormalizationCallback


class TilesExport(NormalizationCallback):
    def __init__(
        self, output_dir: str | Path, normalization_config: DictConfig
    ) -> None:
        super().__init__(normalization_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:  # type: ignore[override]  # intentional: PIL Image is not subtype of ndarray
        return Image.fromarray(super().tensor_to_image(tensor))

    def on_test_batch_end(  # type: ignore[override]  # narrowed Lightning STEP_OUTPUT
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list[torch.Tensor],
        batch: tuple[torch.Tensor, list[dict[str, Any]]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, data = batch
        for b in range(len(outputs)):
            slide_name = data[b]["slide_name"]
            xy = data[b]["xy"]

            slide_dir = self.output_dir / slide_name
            slide_dir.mkdir(parents=True, exist_ok=True)

            self.tensor_to_image(outputs[b]).save(slide_dir / f"{xy}_predicted.png")

            original_image = Image.fromarray(data[b]["original_image"].astype("uint8"))
            original_image.save(slide_dir / f"{xy}_original.png")

            modified_image = Image.fromarray(
                (data[b]["modified_image"] * 255).astype("uint8")
            )
            modified_image.save(slide_dir / f"{xy}_modified.png")

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list[torch.Tensor],
        batch: tuple[torch.Tensor, list[dict[str, Any]]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, data = batch
        for b in range(len(outputs)):
            slide_name = data[b]["slide_name"]
            xy = data[b]["xy"]

            slide_dir = self.output_dir / slide_name
            slide_dir.mkdir(parents=True, exist_ok=True)

            self.tensor_to_image(outputs[b]).save(slide_dir / f"{xy}.png")
