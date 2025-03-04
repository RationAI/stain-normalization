import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDecoder(nn.Module):
    def __init__(self, in_channels=2048, out_channels=3):
        super(UNetDecoder, self).__init__()

        # First upsampling block
        self.upconv1 = nn.ConvTranspose2d(in_channels, 1024, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        # Second upsampling block
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # Third upsampling block
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Fourth upsampling block
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        # Final upsampling to match 512x512 resolution
        self.upconv5 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(8, out_channels, kernel_size=1)  # 3-channel RGB output

    def forward(self, x):
        x = F.relu(self.conv1(self.upconv1(x)))
        x = F.relu(self.conv2(self.upconv2(x)))
        x = F.relu(self.conv3(self.upconv3(x)))
        x = F.relu(self.conv4(self.upconv4(x)))
        x = self.conv5(self.upconv5(x))  # No activation; apply outside
        return x
