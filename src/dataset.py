"""
Dataset for GCP marker images with on-the-fly heatmap target generation.

Each item produces a tuple of:
    (image_tensor, heatmap, class_idx, normalized_xy, rel_path)
"""
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_heatmap(x_norm: float, y_norm: float, size: int = 128, sigma: float = 2.0) -> np.ndarray:
    """
    Generate a 2D Gaussian heatmap of shape (size, size) with peak at (x_norm, y_norm).

    Args:
        x_norm, y_norm: keypoint location in [0, 1] image-relative coordinates.
        size:           heatmap spatial dimension.
        sigma:          Gaussian std-dev in heatmap pixels.
    """
    cx = x_norm * (size - 1)
    cy = y_norm * (size - 1)
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


class GCPDatasetHM(Dataset):
    """
    Args:
        items: list of tuples (rel_path, x_pixel, y_pixel, shape_class_idx).
        root:  base directory; rel_path is joined onto root.
        img_size: input image side after resize.
        hm_size:  output heatmap side.
        train:    if True, applies augmentation (h-flip, color jitter).
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, items, root: str,
                 img_size: int = 512, hm_size: int = 128,
                 train: bool = True):
        self.items = items
        self.root = root
        self.img_size = img_size
        self.hm_size = hm_size
        self.train = train
        self.normalize = transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel, x, y, cls = self.items[idx]
        img = Image.open(os.path.join(self.root, rel)).convert('RGB')
        w, h = img.size

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Per-image normalization. Critical because images vary in resolution
        # (4096x3068, 4096x2730, etc. — NOT the documented 2048x1365).
        nx = max(0.0, min(1.0, x / w))
        ny = max(0.0, min(1.0, y / h))

        arr = np.array(img, dtype=np.float32) / 255.0

        if self.train:
            # Horizontal flip (must also flip x-coord for heatmap)
            if random.random() < 0.5:
                arr = arr[:, ::-1, :].copy()
                nx = 1.0 - nx
            # Light brightness jitter
            if random.random() < 0.5:
                arr = np.clip(arr * random.uniform(0.85, 1.15), 0, 1)

        hm = make_heatmap(nx, ny, size=self.hm_size)
        tensor = self.normalize(torch.from_numpy(arr).permute(2, 0, 1))

        return (
            tensor,
            torch.from_numpy(hm),
            torch.tensor(cls, dtype=torch.long),
            torch.tensor([nx, ny], dtype=torch.float32),
            rel,
        )
