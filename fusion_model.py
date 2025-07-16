import torch
import torch.nn as nn

class FusionNet(nn.Module):
    """
    AMMF-Net: Encoder-Decoder architecture for multimodal medical image fusion.
    Input: two images (CT, MRI), Output: fused image
    """
    def __init__(self):
        super().__init__()
        # Encoder: extract features from concatenated CT and MRI images
        self.enc = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
        )
        # Channel Attention: enhance informative features
        self.att = nn.Sequential(
            nn.Conv2d(128, 128, 1), nn.Sigmoid()
        )
        # Decoder: reconstruct fused image from features
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, ct, mri):
        # Concatenate CT and MRI images along channel dimension
        x = torch.cat([ct, mri], dim=1)  # [B, 2, H, W]
        feat = self.enc(x)
        # Apply channel attention
        feat = feat * self.att(feat)
        # Decode to fused image
        out = self.dec(feat)
        return out