from typing import Any

import numpy as np
import torch
from lightning import Callback


class ImageCallback(Callback):
    """Base callback providing tensor-to-image conversion.

    Expects denormalized [0,1] tensors (denormalization is done in the model).
    """

    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
        """Convert [0,1] CHW tensor to uint8 HWC numpy array."""
        return tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
