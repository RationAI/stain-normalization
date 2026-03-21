import numpy as np
import torch
from kornia.color import rgb_to_lab
from rationai.staining import estimate_stain_vectors
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.image import peak_signal_noise_ratio

from stain_normalization.metrics.vector_metrics import compare_vectors


def _tensor_to_uint8(tensor: Tensor) -> np.ndarray:
    """Convert CHW [0,1] tensor to HWC uint8 numpy array."""
    return (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


class MeanStainDistance(Metric):
    """Mean CIE76 Delta E for hematoxylin and eosin stain vectors.

    Expects denormalized [0,1] tensors. Converts to numpy for stain vector estimation.
    """

    h_sum: Tensor
    h_count: Tensor
    e_sum: Tensor
    e_count: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("h_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("h_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("e_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("e_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        for i in range(preds.shape[0]):
            pred_np = _tensor_to_uint8(preds[i])
            target_np = _tensor_to_uint8(target[i])

            pred_vecs = estimate_stain_vectors(pred_np)
            target_vecs = estimate_stain_vectors(target_np)

            result = compare_vectors(target_vecs, pred_vecs)

            if not np.isnan(result["d_hematoxylin"]):
                self.h_sum += result["d_hematoxylin"]
                self.h_count += 1
            if not np.isnan(result["d_eosin"]):
                self.e_sum += result["d_eosin"]
                self.e_count += 1

    def compute(self) -> dict[str, Tensor]:
        return {
            "d_hematoxylin": self.h_sum / self.h_count
            if self.h_count > 0
            else torch.tensor(float("nan")),
            "d_eosin": self.e_sum / self.e_count
            if self.e_count > 0
            else torch.tensor(float("nan")),
        }


class MeanBrightness(Metric):
    """Mean L* brightness in CIE Lab color space.

    Expects denormalized [0,1] tensors. Uses kornia for GPU-based RGB to Lab conversion.
    """

    brightness_sum: Tensor
    count: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "brightness_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        lab = rgb_to_lab(preds.clamp(0, 1))
        self.brightness_sum += lab[:, 0].mean()
        self.count += 1

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.brightness_sum / self.count


class MeanLabPSNR(Metric):
    """Mean PSNR on the L* channel in CIE Lab color space.

    Expects denormalized [0,1] tensors. Uses kornia for RGB to Lab conversion,
    torchmetrics for PSNR computation.
    """

    psnr_sum: Tensor
    count: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("psnr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        pred_lab = rgb_to_lab(preds.clamp(0, 1))
        target_lab = rgb_to_lab(target.clamp(0, 1))

        pred_l = pred_lab[:, 0:1]
        target_l = target_lab[:, 0:1]

        self.psnr_sum += peak_signal_noise_ratio(pred_l, target_l, data_range=100.0)
        self.count += 1

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.psnr_sum / self.count


class MeanPCC(Metric):
    """Mean Pearson Correlation Coefficient between image pairs.

    Operates on raw tensors (no denormalization needed).
    """

    pcc_sum: Tensor
    count: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("pcc_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        for i in range(preds.shape[0]):
            x = preds[i].flatten().float()
            y = target[i].flatten().float()

            if x.std() == 0 or y.std() == 0:
                continue

            pcc = torch.corrcoef(torch.stack([x, y]))[0, 1]
            self.pcc_sum += pcc
            self.count += 1

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.pcc_sum / self.count
