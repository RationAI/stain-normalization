"""
The SSIM is based on implementation from gaussian-splatting and slightly simplified
(pre-computed windows and removal of unused arguments).
https://github.com/graphdeco-inria/gaussian-splatting/blob/472689c0dc70417448fb451bf529ae532d32c095/utils/loss_utils.py
"""

from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1SSIMLoss(nn.Module):
    def __init__(
        self,
        lambda_dssim: float = 0.6,
        lambda_l1: float = 0.2,
        lambda_lum: float = 0.2,
        lambda_gdl: float = 0.1,
    ):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.lambda_l1 = lambda_l1
        self.lambda_lum = lambda_lum
        self.lambda_gdl = lambda_gdl

        # precompute SSIM windows to avoid repetition
        self.window_size = 11
        self.channel = 3
        self._1d_window = gaussian(self.window_size, 1.5).unsqueeze(1)
        self._2d_window = (
            self._1d_window.mm(self._1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        )
        self.window: torch.Tensor
        self.register_buffer(
            "window",
            self._2d_window.expand(
                self.channel, 1, self.window_size, self.window_size
            ).contiguous(),
        )

    def forward(self, image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        # L1 color loss
        l1_loss = F.l1_loss(image, target_image, reduction="mean")

        # SSIM structural loss
        ssim_loss = 1.0 - self._ssim(image, target_image, self.window)

        # Gradient loss for edges
        gdl_loss = gradient_loss(image, target_image)

        # Luminance / brightness loss
        brig_loss = brightness_loss(image, target_image)

        # total weighted loss
        total_loss = (
            self.lambda_l1 * l1_loss
            + self.lambda_dssim * ssim_loss
            + self.lambda_gdl * gdl_loss
            + self.lambda_lum * brig_loss
        )

        return total_loss

    @torch.compile
    def _ssim(
        self, img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor
    ) -> torch.Tensor:
        # Modified _ssim that uses pre-computed window
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(
                img1 * img1, window, padding=self.window_size // 2, groups=self.channel
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                img2 * img2, window, padding=self.window_size // 2, groups=self.channel
            )
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                img1 * img2, window, padding=self.window_size // 2, groups=self.channel
            )
            - mu1_mu2
        )

        c1 = 0.01**2
        c2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )

        return ssim_map.mean()


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def brightness_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean(dim=[1, 2, 3])
    target_mean = target.mean(dim=[1, 2, 3])
    return F.l1_loss(pred_mean, target_mean)


def gradient_loss(image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
    def gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])  # Horizontal gradient
        dy = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])  # Vertical gradient
        return dx, dy

    image_dx, image_dy = gradient(image)
    target_dx, target_dy = gradient(target_image)

    loss_x = F.l1_loss(image_dx, target_dx, reduction="mean")
    loss_y = F.l1_loss(image_dy, target_dy, reduction="mean")

    return loss_x + loss_y
