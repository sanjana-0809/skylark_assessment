"""
Multi-task heatmap-based model for GCP marker localization and classification.

Architecture: ResNet18 encoder + UNet-style decoder for keypoint heatmap regression,
plus a parallel classification head from the deepest feature map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HeatmapModel(nn.Module):
    """
    Multi-task model with shared ResNet18 backbone.

    Outputs:
        heatmap:    [B, 128, 128] single-channel heatmap (1/4 of 512 input size)
        cls_logits: [B, num_classes] classification logits
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        bb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Encoder: split ResNet18 into stages so we can grab skip-connection features
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1   # 64 ch,  1/4 spatial
        self.layer2 = bb.layer2   # 128 ch, 1/8
        self.layer3 = bb.layer3   # 256 ch, 1/16
        self.layer4 = bb.layer4   # 512 ch, 1/32

        # Decoder: upsample and fuse with encoder features (UNet pattern)
        self.up3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)

        # Classification head: GAP on deepest features
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x0 = self.stem(x)         # [B, 64, 128, 128]
        x1 = self.layer1(x0)      # [B, 64, 128, 128]
        x2 = self.layer2(x1)      # [B, 128, 64, 64]
        x3 = self.layer3(x2)      # [B, 256, 32, 32]
        x4 = self.layer4(x3)      # [B, 512, 16, 16]

        u3 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = self.up3(torch.cat([u3, x3], dim=1))     # [B, 256, 32, 32]

        u2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, x2], dim=1))     # [B, 128, 64, 64]

        u1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, x1], dim=1))     # [B, 64, 128, 128]

        heatmap = self.heatmap_head(u1).squeeze(1)    # [B, 128, 128]
        cls_logits = self.cls_head(x4)                # [B, num_classes]
        return heatmap, cls_logits


def soft_argmax_2d(hm: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
    """
    Differentiable, sub-pixel-precise argmax via softmax-weighted spatial average.

    Args:
        hm:   [B, H, W] heatmap tensor (raw logits or probabilities).
        beta: temperature. Higher = sharper peak. 100 works well for trained models.

    Returns:
        [B, 2] tensor of normalized (x, y) coordinates in [0, 1].
    """
    B, H, W = hm.shape
    flat = hm.view(B, -1) * beta
    soft = F.softmax(flat, dim=1).view(B, H, W)

    ys = torch.arange(H, device=hm.device, dtype=torch.float32).view(1, H, 1)
    xs = torch.arange(W, device=hm.device, dtype=torch.float32).view(1, 1, W)

    cy = (soft * ys).sum(dim=(1, 2)) / (H - 1)
    cx = (soft * xs).sum(dim=(1, 2)) / (W - 1)
    return torch.stack([cx, cy], dim=1)
