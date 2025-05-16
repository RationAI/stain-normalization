from pathlib import Path
from lightning import Callback, LightningModule, Trainer
from PIL import Image
from omegaconf import DictConfig
import torch


class TilesExport(Callback):
    def __init__(self, output_dir: str | Path, normalization_config: DictConfig) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        normalization = normalization_config
        self.mean = torch.tensor(normalization.mean).view(3, 1, 1)
        self.std = torch.tensor(normalization.std).view(3, 1, 1)
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        device = tensor.device
        std = self.std.to(device)
        mean = self.mean.to(device)
        return (tensor * std) + mean
        
    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        tensor = self.denormalize(tensor)  
        tensor = tensor.clamp(0, 1)  
        tensor = (tensor * 255).byte()
        return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())  
    
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list[torch.Tensor],
        batch: tuple[torch.Tensor, list],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._save_images(outputs, batch, is_predict=False)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list[torch.Tensor],
        batch: tuple[torch.Tensor, list],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._save_images(outputs, batch, is_predict=True)

    def _save_images(self, outputs, batch, is_predict: bool) -> None:
        inputs, data = batch
        for b in range(len(outputs)):
            index = data[b].get("index", 0)

            predicted_image = outputs[b]
            name = data[b].get("name", str(index))
            predicted_image = self.tensor_to_image(predicted_image)

            predicted_image.save(self.output_dir / f"{name}.png")

            if not is_predict:
                original_image = Image.fromarray(data[b]["original_image"].astype("uint8"))
                original_image.save(self.output_dir / f"{name}_original.png")

                if "modified_image" in data[b]:
                    modified_image = Image.fromarray((data[b]["modified_image"] * 255).astype("uint8"))
                    modified_image.save(self.output_dir / f"{name}_modified.png")
