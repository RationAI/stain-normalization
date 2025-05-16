from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig
from PIL import Image


class SaveWSI(Callback):
    def __init__(self, output_dir: str | Path, normalization_config: DictConfig) -> None:
        super().__init__()
        self.output_root = Path(output_dir)
        self.mean = torch.tensor(normalization_config.mean).view(3, 1, 1)
        self.std = torch.tensor(normalization_config.std).view(3, 1, 1)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        device = tensor.device
        std = self.std.to(device)
        mean = self.mean.to(device)
        return (tensor * std) + mean

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs: list[torch.Tensor],
        batch: tuple[torch.Tensor, list],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        
        for b in range(len(outputs)):
            predicted_image = outputs[b]
            metadata = batch[1][b]
    
            slide_name = metadata["slide_name"]
            level = metadata["level"]
            original_image = Image.fromarray(metadata["original_image"].astype("uint8"))
            filename = f'{metadata["xy"]}.png'

            base_folder = self.output_root / str(slide_name) / str(level)
            predicted_folder = base_folder / "predicted"
            original_folder = base_folder / "original"

            predicted_folder.mkdir(parents=True, exist_ok=True)
            original_folder.mkdir(parents=True, exist_ok=True)

            original_image.save(original_folder / filename)

            predicted_image = self.denormalize(predicted_image).clamp(0, 1)
            predicted_image = (predicted_image * 255).byte()
            predicted_image = Image.fromarray(predicted_image.permute(1, 2, 0).cpu().numpy())
            predicted_image.save(predicted_folder / filename)
