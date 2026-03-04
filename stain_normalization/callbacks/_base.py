import numpy as np
import torch
from lightning import Callback
from omegaconf import DictConfig


class NormalizationCallback(Callback):
    """Base callback providing denormalization helpers for model outputs."""

    def __init__(self, normalization_config: DictConfig) -> None:
        super().__init__()
        self.mean = torch.tensor(normalization_config.mean).view(3, 1, 1)
        self.std = torch.tensor(normalization_config.std).view(3, 1, 1)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverse normalization: tensor → [0, 1] float."""
        device = tensor.device
        return (tensor * self.std.to(device)) + self.mean.to(device)

    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output tensor to uint8 HWC numpy array."""
        return (
            self.denormalize(tensor).clamp(0, 1).mul(255).byte()
            .permute(1, 2, 0).cpu().numpy()
        )
