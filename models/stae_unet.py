
import torch
import torch.nn as nn
from models.transformer_block import TransformerBlock

class STAEUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc2 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1)
        self.transformer = TransformerBlock(base_channels*4, n_heads=4)
        self.up1 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec1 = nn.Conv3d(base_channels*4, base_channels*2, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec2 = nn.Conv3d(base_channels*2, base_channels, kernel_size=3, padding=1)
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = torch.relu(self.enc1(x))
        x2 = torch.relu(self.enc2(x1))
        x3 = torch.relu(self.enc3(x2))

        B, C, D, H, W = x3.shape
        x3_flat = x3.view(B, C, -1).transpose(1, 2)
        x3_trans = self.transformer(x3_flat).transpose(1, 2).view(B, C, D, H, W)

        u1 = self.up1(x3_trans)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = torch.relu(self.dec1(u1))

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = torch.relu(self.dec2(u2))

        return torch.sigmoid(self.out_conv(u2))
