"""Original SSIM code based on pytorch-ssim by Evan Su (MIT License).

https://github.com/Po-Hsun-Su/pytorch-ssim .
"""

from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class L1SSIMLoss(nn.Module):
    def __init__(self, lambda_dssim: float = 0.6, lambda_gdl: float = 0.2):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.lambda_gdl = lambda_gdl

    def forward(self, image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(image, target_image, reduction="mean")
        ssim_loss = 1.0 - ssim(image, target_image)

        gdl_loss = gradient_loss(image, target_image)  # GDL for sharp edges

        total_loss = (
            (1.0 - self.lambda_dssim) * l1
            + self.lambda_dssim * ssim_loss
            + self.lambda_gdl * gdl_loss
        )

        return total_loss


def gradient_loss(image, target_image):
    def gradient(x):
        dx = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])  # Horizontal gradient
        dy = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])  # Vertical gradient
        return dx, dy

    image_dx, image_dy = gradient(image)
    target_dx, target_dy = gradient(target_image)

    loss_x = F.l1_loss(image_dx, target_dx, reduction="mean")
    loss_y = F.l1_loss(image_dy, target_dy, reduction="mean")

    return loss_x + loss_y


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1d_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
