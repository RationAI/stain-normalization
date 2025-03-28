import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from math import exp
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


class L1SSIMLoss(nn.Module):  
    def __init__(self, lambda_dssim: float = 0.6, lambda_gdl: float = 0.2, ):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.lambda_gdl = lambda_gdl

    def forward(self, image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        Ll1 = F.l1_loss(image, target_image, reduction='mean')  # L1 loss
        ssim_loss = 1.0 - SSIM(image, target_image)  

        gdl_loss = gradient_loss(image, target_image)  # GDL for sharp edges
        # hf_loss = high_frequency_loss(image, target_image)  # HF loss for textures

        total_loss = (1.0 - self.lambda_dssim) * Ll1 + \
                     self.lambda_dssim * ssim_loss + \
                     self.lambda_gdl * gdl_loss 
                     
        return total_loss


def gradient_loss(image, target_image):
    def gradient(x):
        dx = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])  # Horizontal gradient
        dy = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])  # Vertical gradient
        return dx, dy

    image_dx, image_dy = gradient(image)
    target_dx, target_dy = gradient(target_image)

    loss_x = F.l1_loss(image_dx, target_dx, reduction='mean')
    loss_y = F.l1_loss(image_dy, target_dy, reduction='mean')

    return loss_x + loss_y

