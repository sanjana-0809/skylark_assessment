"""
Train the multi-task heatmap model.

Usage:
    python src/train.py \
        --train-dir path/to/train_dataset \
        --labels    path/to/train_dataset/gcp_marks.json \
        --epochs    12 \
        --batch     16 \
        --out       best_model.pt
"""
import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import GCPDatasetHM
from model import HeatmapModel, soft_argmax_2d

SHAPE_CLASSES = ['Cross', 'Square', 'L-Shape']
SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPE_CLASSES)}


def build_items(labels_path: str, root: str):
    """Load labels JSON, filter for valid samples, return list of (rel, x, y, cls_idx)."""
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    items = []
    skipped = 0
    for rel, v in labels.items():
        if not os.path.exists(os.path.join(root, rel)):
            skipped += 1
            continue
        shape = v['verified_shape']
        if shape not in SHAPE_TO_IDX:
            skipped += 1
            continue
        items.append((rel, v['mark']['x'], v['mark']['y'], SHAPE_TO_IDX[shape]))

    print(f"Loaded {len(items)} valid samples (skipped {skipped})")
    return items


def evaluate(model, loader, device, val_sizes, kp_loss_fn, ce_loss_fn, kp_weight):
    model.eval()
    n = 0
    v_loss = 0.0
    v_correct = 0
    v_p10 = v_p25 = v_p50 = 0
    with torch.no_grad():
        for imgs, hms, cls, kps, rels in loader:
            imgs, hms, cls, kps = imgs.to(device), hms.to(device), cls.to(device), kps.to(device)
            pred_hm, pred_cls = model(imgs)
            loss = kp_loss_fn(pred_hm, hms) * kp_weight + ce_loss_fn(pred_cls, cls)
            v_loss += loss.item() * imgs.size(0)
            v_correct += (pred_cls.argmax(1) == cls).sum().item()

            pred_xy = soft_argmax_2d(pred_hm)
            for i, r in enumerate(rels):
                w_i, h_i = val_sizes[r]
                dx = (pred_xy[i, 0].item() - kps[i, 0].item()) * w_i
                dy = (pred_xy[i, 1].item() - kps[i, 1].item()) * h_i
                d = (dx * dx + dy * dy) ** 0.5
                if d < 10: v_p10 += 1
                if d < 25: v_p25 += 1
                if d < 50: v_p50 += 1
            n += imgs.size(0)

    return {
        'loss': v_loss / n,
        'acc': v_correct / n,
        'pck10': v_p10 / n,
        'pck25': v_p25 / n,
        'pck50': v_p50 / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--img-size', type=int, default=512)
    ap.add_argument('--out', default='best_model.pt')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data ---
    items = build_items(args.labels, args.train_dir)
    train_items, val_items = train_test_split(
        items, test_size=0.15, random_state=args.seed,
        stratify=[it[3] for it in items]
    )
    print(f"Train: {len(train_items)}  Val: {len(val_items)}")

    train_ds = GCPDatasetHM(train_items, args.train_dir, img_size=args.img_size, train=True)
    val_ds = GCPDatasetHM(val_items, args.train_dir, img_size=args.img_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Cache val image dimensions for accurate PCK
    print("Caching validation image sizes...")
    val_sizes = {rel: Image.open(os.path.join(args.train_dir, rel)).size
                 for rel, _, _, _ in val_items}

    # --- Model ---
    model = HeatmapModel(num_classes=len(SHAPE_CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f}M")

    # --- Loss & optimizer ---
    counts = Counter(it[3] for it in train_items)
    class_weights = torch.tensor(
        [1.0 / counts[i] for i in range(len(SHAPE_CLASSES))],
        dtype=torch.float32, device=device,
    )
    class_weights = class_weights / class_weights.sum() * len(SHAPE_CLASSES)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    encoder_params = (list(model.stem.parameters()) +
                      list(model.layer1.parameters()) + list(model.layer2.parameters()) +
                      list(model.layer3.parameters()) + list(model.layer4.parameters()))
    decoder_params = (list(model.up3.parameters()) + list(model.up2.parameters()) +
                      list(model.up1.parameters()) +
                      list(model.heatmap_head.parameters()) +
                      list(model.cls_head.parameters()))

    opt = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 1e-4},
        {'params': decoder_params, 'lr': 1e-3},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss(weight=class_weights)
    KP_W = 1000.0   # heatmap MSE values are tiny; this balances against CE

    # --- Training loop ---
    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, hms, cls, _, _ in train_loader:
            imgs, hms, cls = imgs.to(device), hms.to(device), cls.to(device)
            pred_hm, pred_cls = model(imgs)
            loss = mse(pred_hm, hms) * KP_W + ce(pred_cls, cls)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item() * imgs.size(0)
        sched.step()
        train_loss = running / len(train_ds)

        m = evaluate(model, val_loader, device, val_sizes, mse, ce, KP_W)
        print(f"Ep {epoch+1:>2}/{args.epochs}  train={train_loss:.4f}  val={m['loss']:.4f}  "
              f"acc={m['acc']:.3f}  PCK10={m['pck10']:.3f}  PCK25={m['pck25']:.3f}  "
              f"PCK50={m['pck50']:.3f}")

        if m['loss'] < best_val:
            best_val = m['loss']
            torch.save(model.state_dict(), args.out)
            print(f"  ✓ saved best model to {args.out}")

    print("Training complete.")


if __name__ == '__main__':
    main()
