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


class _DenormalizingMetric(Metric):
    """Base class for metrics that need denormalized [0,1] images.

    If denormalize_mean/std are None, assumes input is already in [0,1] range.
    """

    def __init__(
        self,
        denormalize_mean: list[float] | None = None,
        denormalize_std: list[float] | None = None,
    ) -> None:
        super().__init__()
        if denormalize_mean is not None and denormalize_std is not None:
            self.register_buffer(
                "denorm_mean", torch.tensor(denormalize_mean).view(3, 1, 1)
            )
            self.register_buffer(
                "denorm_std", torch.tensor(denormalize_std).view(3, 1, 1)
            )
        else:
            self.register_buffer("denorm_mean", None)
            self.register_buffer("denorm_std", None)

    def _denormalize(self, tensor: Tensor) -> Tensor:
        if self.denorm_mean is None:
            return tensor
        mean: Tensor = self.denorm_mean  # type: ignore[assignment]
        std: Tensor = self.denorm_std  # type: ignore[assignment]
        return tensor * std.to(tensor.device) + mean.to(tensor.device)


# --- Stain vector metrics (numpy-based) ---


class _BaseStainDistance(_DenormalizingMetric):
    """Base class for stain vector distance metrics.

    Converts tensors to numpy for stain vector estimation.
    """

    distance_sum: Tensor
    count: Tensor

    def __init__(
        self,
        denormalize_mean: list[float] | None = None,
        denormalize_std: list[float] | None = None,
    ) -> None:
        super().__init__(denormalize_mean, denormalize_std)
        self.add_state("distance_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _stain_key(self) -> str:
        raise NotImplementedError

    def update(self, preds: Tensor, target: Tensor) -> None:
        key = self._stain_key()
        for i in range(preds.shape[0]):
            pred_np = _tensor_to_uint8(self._denormalize(preds[i]))
            target_np = _tensor_to_uint8(self._denormalize(target[i]))

            pred_vecs = estimate_stain_vectors(pred_np)
            target_vecs = estimate_stain_vectors(target_np)

            result = compare_vectors(target_vecs, pred_vecs)

            if not np.isnan(result[key]):
                self.distance_sum += result[key]
                self.count += 1

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.distance_sum / self.count


class MeanHematoxylinDistance(_BaseStainDistance):
    """Mean CIE76 Delta E for hematoxylin stain vectors."""

    def _stain_key(self) -> str:
        return "d_hematoxylin"


class MeanEosinDistance(_BaseStainDistance):
    """Mean CIE76 Delta E for eosin stain vectors."""

    def _stain_key(self) -> str:
        return "d_eosin"


class MeanBrightness(_DenormalizingMetric):
    """Mean L* brightness in CIE Lab color space.

    Uses kornia for GPU-based RGB to Lab conversion.
    """

    brightness_sum: Tensor
    count: Tensor

    def __init__(
        self,
        denormalize_mean: list[float] | None = None,
        denormalize_std: list[float] | None = None,
    ) -> None:
        super().__init__(denormalize_mean, denormalize_std)
        self.add_state(
            "brightness_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        denormed = self._denormalize(preds).clamp(0, 1)
        lab = rgb_to_lab(denormed)
        self.brightness_sum += lab[:, 0].mean()
        self.count += 1

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.brightness_sum / self.count


class MeanLabPSNR(_DenormalizingMetric):
    """Mean PSNR on the L* channel in CIE Lab color space.

    Uses kornia for GPU-based RGB to Lab conversion,
    torchmetrics for PSNR computation.
    """

    psnr_sum: Tensor
    count: Tensor

    def __init__(
        self,
        denormalize_mean: list[float] | None = None,
        denormalize_std: list[float] | None = None,
    ) -> None:
        super().__init__(denormalize_mean, denormalize_std)
        self.add_state("psnr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        pred_lab = rgb_to_lab(self._denormalize(preds).clamp(0, 1))
        target_lab = rgb_to_lab(self._denormalize(target).clamp(0, 1))

        # L channel: BxHxW, data_range=100.0
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

    Operates on raw normalized tensors (no denormalization needed).
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
