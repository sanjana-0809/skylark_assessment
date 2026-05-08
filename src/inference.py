"""
Run inference on a directory of test images and write predictions.json.

Usage:
    python src/inference.py \
        --weights  path/to/best_model.pt \
        --test-dir path/to/test_dataset \
        --out      predictions.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import HeatmapModel, soft_argmax_2d

SHAPE_CLASSES = ['Cross', 'Square', 'L-Shape']
IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_images(test_dir: str):
    paths = []
    for ext in ('*.JPG', '*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.PNG'):
        paths += glob.glob(os.path.join(test_dir, '**', ext), recursive=True)
    return sorted(paths)


def predict(model, test_dir: str, device, batch: int = 16):
    paths = find_images(test_dir)
    print(f"Found {len(paths)} test images")

    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    predictions = {}
    buf_imgs, buf_rels, buf_sizes = [], [], []

    def flush():
        if not buf_imgs:
            return
        x = torch.stack(buf_imgs).to(device)
        with torch.no_grad():
            pred_hm, pred_cls = model(x)
            pred_xy = soft_argmax_2d(pred_hm).cpu().numpy()
            ci = pred_cls.argmax(1).cpu().numpy()
        for i, rel in enumerate(buf_rels):
            w_i, h_i = buf_sizes[i]
            predictions[rel] = {
                "mark": {
                    "x": round(float(pred_xy[i, 0]) * w_i, 2),
                    "y": round(float(pred_xy[i, 1]) * h_i, 2),
                },
                "verified_shape": SHAPE_CLASSES[int(ci[i])],
            }

    for p in paths:
        rel = os.path.relpath(p, test_dir).replace(os.sep, '/')
        img = Image.open(p).convert('RGB')
        buf_sizes.append(img.size)
        arr = np.array(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        t = normalize(torch.from_numpy(arr).permute(2, 0, 1))
        buf_imgs.append(t)
        buf_rels.append(rel)
        if len(buf_imgs) >= batch:
            flush()
            buf_imgs, buf_rels, buf_sizes = [], [], []
    flush()

    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='Path to best_model.pt')
    ap.add_argument('--test-dir', required=True, help='Path to test_dataset directory')
    ap.add_argument('--out', default='predictions.json')
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = HeatmapModel(num_classes=len(SHAPE_CLASSES)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    preds = predict(model, args.test_dir, device, batch=args.batch)

    with open(args.out, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f"\n✓ Wrote {len(preds)} predictions to {args.out}")


if __name__ == '__main__':
    main()
