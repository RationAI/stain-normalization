from pathlib import Path
from lightning import Callback, LightningModule, Trainer
from PIL import Image
from omegaconf import DictConfig
import torch


class TilesExport(Callback):
    def __init__(self, output_dir: str | Path, predict_normalization_config: DictConfig) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract normalization parameters from the config (predict normalization)
        normalization = predict_normalization_config
        self.mean = torch.tensor(normalization.mean).view(3, 1, 1)
        self.std = torch.tensor(normalization.std).view(3, 1, 1)


    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        device = tensor.device  
            
        std = self.std.to(device)
        mean = self.mean.to(device)
        
        return (tensor * std) + mean

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a PyTorch tensor to a PIL image ."""
        tensor = self.denormalize(tensor)  
        tensor = tensor.clamp(0, 1)  
        tensor = (tensor * 255).byte()
        return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())  

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list[torch.Tensor],  
        batch: tuple[torch.Tensor, list],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Saves three images per sample:
        - Original Image (ground truth)
        - Modified Image (input to model)
        - Predicted Image (model output)
        """
        for b in range(len(outputs)):
            predicted_image = outputs[b]
            original_image = Image.fromarray(batch[1][b]["original_image"].astype("uint8"))
            modified_image = Image.fromarray((batch[1][b]["modified_image"] * 255).astype("uint8"))
            index = batch[1][b]["index"]

            predicted_image = self.tensor_to_image(predicted_image)

            # Save images
            original_image.save(self.output_dir / f"{index}_original.png")
            modified_image.save(self.output_dir / f"{index}_modified.png")
            predicted_image.save(self.output_dir / f"{index}_predicted.png")

