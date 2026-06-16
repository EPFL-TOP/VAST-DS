"""Multi-head training for the profile_v2 model.

Reads ``manifest_v2.csv`` from ``export_training_set_v2`` and trains a
ResNet18 backbone with two heads:

  * **Severity head** — 5-class classifier (healthy / mild / moderate /
    severe / reject). The 'reject' class lets us drop false-positive
    candidates from a looser profile_v1, which is the only way to close
    the missing-tail-somite gap.

  * **Bbox-regression head** — 4 outputs ``(cx, cy, w, h)`` in patch
    coordinates (sigmoid → [0, 1]). Smooth-L1 against the annotator's
    labelled bbox. Loss is masked out for 'reject' rows (no real somite).

Total loss = weighted CE(severity) + λ · smooth_L1(bbox), with sqrt-inv
class weights for severity (same scheme as ``training_severity.py``).

Split is **by fish** (``dest_well_id``) — never the same well in train
and val/test. Augmentation = horizontal flip; bbox cx is mirrored
accordingly when flipped.

Outputs:
  * ``checkpoints/profile_v2_best.pth``
  * ``checkpoints/profile_v2_test_metrics.json``  (per-class P/R/F1 +
    confusion matrix + mean L1 bbox error in patch-fraction units)

Examples
--------
    python -m somiteCounting.training_profile_v2 \\
        --manifest data/profile_v2_training_set/manifest_v2.csv

    python -m somiteCounting.training_profile_v2 \\
        --manifest data/profile_v2_training_set/manifest_v2.csv \\
        --output_dir checkpoints --epochs 80 --lambda_bbox 1.5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ResNet18_Weights

# Allow both `python somiteCounting/training_profile_v2.py` and
# `python -m somiteCounting.training_profile_v2`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from somiteCounting._common import _device, preprocess_image, train_loop  # noqa: E402


CLASS_LABELS = ['healthy', 'mild', 'moderate', 'severe', 'reject']
NUM_CLASSES = 5
PATCH_DEFAULT = 224


# =====================================================================
# Model
# =====================================================================
class ProfileV2Head(nn.Module):
    """Two heads sharing the ResNet backbone's pooled feature vector."""

    def __init__(self, in_features: int, num_severity: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.severity_fc = nn.Linear(in_features, num_severity)
        self.bbox_fc = nn.Linear(in_features, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(x)
        sev_logits = self.severity_fc(x)
        # Sigmoid keeps bbox outputs in [0, 1] — the natural range for
        # normalised patch coordinates. Smooth-L1 against the targets.
        bbox_preds = torch.sigmoid(self.bbox_fc(x))
        return sev_logits, bbox_preds


def make_profile_v2_model(*, num_severity: int = NUM_CLASSES,
                          dropout: float = 0.3) -> nn.Module:
    """ResNet18 (ImageNet) adapted to 1-channel input, with the two-head
    output. Conv1's RGB weights are averaged into a single grayscale
    channel — same trick as ``make_grayscale_resnet18``."""
    base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    old_conv1 = base.conv1
    base.conv1 = nn.Conv2d(
        1, old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=False,
    )
    with torch.no_grad():
        avg = old_conv1.weight.mean(dim=1, keepdim=True)
        base.conv1.weight.copy_(avg)
    in_features = base.fc.in_features
    base.fc = ProfileV2Head(in_features, num_severity, dropout)
    return base


# =====================================================================
# Dataset
# =====================================================================
class ProfileV2Dataset(Dataset):
    """(image, severity_class, target_bbox, has_bbox) tuples from a
    list of manifest rows."""

    def __init__(self, samples, augment: bool = False,
                 patch_size: int = PATCH_DEFAULT):
        self.samples = list(samples)
        self.augment = augment
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        img_np = np.array(Image.open(row['_path'])).astype(np.float32)

        # Target [cx, cy, w, h] in patch fraction (0..1)
        cx = float(row['target_cx'])
        cy = float(row['target_cy'])
        w  = float(row['target_w'])
        h  = float(row['target_h'])

        # Horizontal-flip augmentation. Mirror cx → 1 - cx.
        if self.augment and np.random.rand() < 0.5:
            img_np = np.ascontiguousarray(np.fliplr(img_np))
            cx = 1.0 - cx

        img = preprocess_image(img_np, resize=(self.patch_size, self.patch_size))
        target_bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)
        severity = torch.tensor(int(row['class_idx']), dtype=torch.long)
        has_bbox = torch.tensor(bool(row['has_bbox_target']), dtype=torch.bool)
        return img, severity, target_bbox, has_bbox


# =====================================================================
# Loss
# =====================================================================
class ProfileV2Loss(nn.Module):
    """Weighted CE on severity + smooth-L1 on bbox (masked for reject)."""

    def __init__(self, sev_weights: torch.Tensor, lambda_bbox: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=sev_weights)
        self.lambda_bbox = lambda_bbox

    def forward(self, preds, severity, target_bbox, has_bbox):
        sev_logits, bbox_preds = preds
        ce = self.ce(sev_logits, severity)
        if has_bbox.any():
            bbox_l = F.smooth_l1_loss(
                bbox_preds[has_bbox], target_bbox[has_bbox])
        else:
            bbox_l = torch.tensor(0.0, device=ce.device)
        return ce + self.lambda_bbox * bbox_l


def _v2_batch_unpacker(batch, device):
    img, severity, target_bbox, has_bbox = batch
    model_inputs = (img.to(device),)
    crit_args = (
        severity.to(device),
        target_bbox.to(device),
        has_bbox.to(device),
    )
    return model_inputs, crit_args


# =====================================================================
# Manifest I/O + splitting
# =====================================================================
def _read_manifest(path: str):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r['class_idx']    = int(r['class_idx'])
                r['dest_well_id'] = int(r['dest_well_id'])
                r['has_bbox_target'] = (str(r['has_bbox_target']).lower()
                                        in ('true', '1', 'yes'))
                r['target_cx']    = float(r['target_cx'])
                r['target_cy']    = float(r['target_cy'])
                r['target_w']     = float(r['target_w'])
                r['target_h']     = float(r['target_h'])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def _split_by_well(rows, *, train_pct=0.70, val_pct=0.15, seed=42):
    by_well = defaultdict(list)
    for r in rows:
        by_well[r['dest_well_id']].append(r)
    wells = sorted(by_well.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(wells)
    n = len(wells)
    n_train = int(round(n * train_pct))
    n_val   = int(round(n * val_pct))
    train_wells = wells[:n_train]
    val_wells   = wells[n_train:n_train + n_val]
    test_wells  = wells[n_train + n_val:]
    return ([r for w in train_wells for r in by_well[w]],
            [r for w in val_wells   for r in by_well[w]],
            [r for w in test_wells  for r in by_well[w]])


def _class_weights_sqrt_inv(samples, num_classes=NUM_CLASSES):
    cnt = Counter(r['class_idx'] for r in samples)
    freqs = np.array([max(cnt.get(c, 0), 1) for c in range(num_classes)],
                     dtype=np.float64)
    w = 1.0 / np.sqrt(freqs)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# =====================================================================
# Evaluation
# =====================================================================
def _evaluate(model, dataset, batch_size=16):
    device = _device()
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    bbox_l1_total = 0.0
    bbox_n = 0
    with torch.no_grad():
        for img, sev, tbox, has_bbox in loader:
            img = img.to(device)
            sev_logits, bbox_pred = model(img)
            preds = sev_logits.argmax(dim=1).cpu().numpy()
            true  = sev.numpy()
            for t, p in zip(true, preds):
                cm[t, p] += 1
            # Bbox L1 only on rows with valid target
            if has_bbox.any():
                err = (bbox_pred[has_bbox].cpu() - tbox[has_bbox]).abs().mean(dim=1)
                bbox_l1_total += float(err.sum())
                bbox_n += int(has_bbox.sum())

    per_class = {}
    for c in range(NUM_CLASSES):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum()) - tp
        fp = int(cm[:, c].sum()) - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        per_class[CLASS_LABELS[c]] = {
            'precision': float(p), 'recall': float(r), 'f1': float(f1),
            'support': int(cm[c, :].sum()),
        }
    n = int(cm.sum())
    acc = float(cm.trace()) / max(n, 1)
    bbox_mae = (bbox_l1_total / max(bbox_n, 1)) if bbox_n else None
    return {
        'n':                n,
        'accuracy':         acc,
        'bbox_mae_patchfrac': bbox_mae,
        'bbox_n':           int(bbox_n),
        'per_class':        per_class,
        'confusion_matrix': cm.tolist(),
        'class_labels':     CLASS_LABELS,
    }


# =====================================================================
# Top-level orchestration
# =====================================================================
def train_profile_v2(*,
                     manifest: str,
                     output_dir: str = 'checkpoints',
                     epochs: int = 50,
                     batch_size: int = 16,
                     patience: int = 8,
                     lr_head: float = 1e-4,
                     lr_backbone: float = 1e-5,
                     dropout: float = 0.3,
                     lambda_bbox: float = 1.0,
                     patch_size: int = PATCH_DEFAULT,
                     seed: int = 42):
    rows = _read_manifest(manifest)
    if not rows:
        raise SystemExit(f"manifest is empty / unreadable: {manifest}")
    print(f"Read {len(rows)} rows from {manifest}")

    # Resolve image paths relative to manifest dir.
    root = os.path.dirname(os.path.abspath(manifest))
    for r in rows:
        r['_path'] = os.path.join(root, r['tile_path'])

    tr, vl, te = _split_by_well(rows, seed=seed)
    print(f"Split (by fish): train={len(tr)} val={len(vl)} test={len(te)} "
          f"(across {len({r['dest_well_id'] for r in rows})} wells)")

    train_cls = Counter(r['class_idx'] for r in tr)
    print(f"Train class distribution: "
          f"{ {CLASS_LABELS[c]: train_cls.get(c, 0) for c in range(NUM_CLASSES)} }")
    if any(train_cls.get(c, 0) < 5 for c in range(NUM_CLASSES) if c != 4):
        print("  WARNING: at least one severity class has <5 train samples — "
              "results will be unreliable. Annotate more and re-run.")

    weights = _class_weights_sqrt_inv(tr)
    print(f"Class weights (1/sqrt(freq)): "
          f"{ {CLASS_LABELS[c]: round(float(weights[c]), 3) for c in range(NUM_CLASSES)} }")

    train_ds = ProfileV2Dataset(tr, augment=True,  patch_size=patch_size)
    val_ds   = ProfileV2Dataset(vl, augment=False, patch_size=patch_size)
    test_ds  = ProfileV2Dataset(te, augment=False, patch_size=patch_size)

    device = _device()
    print(f"Using device: {device}")
    model = make_profile_v2_model(num_severity=NUM_CLASSES, dropout=dropout)
    criterion = ProfileV2Loss(weights.to(device), lambda_bbox=lambda_bbox)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'profile_v2_best.pth')

    model = train_loop(
        model, train_ds, val_ds, criterion,
        save_path=save_path,
        epochs=epochs, batch_size=batch_size,
        lr_head=lr_head, lr_backbone=lr_backbone,
        patience=patience,
        batch_unpacker=_v2_batch_unpacker,
    )

    # Test-set evaluation
    print("\nEvaluating on test set…")
    if not te:
        print("  (no test samples — too few wells; metrics file not written)")
        return model, None

    metrics = _evaluate(model, test_ds, batch_size=batch_size)
    metrics_path = os.path.join(output_dir, 'profile_v2_test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTest set: N={metrics['n']}  accuracy={metrics['accuracy']:.3f}")
    if metrics['bbox_mae_patchfrac'] is not None:
        print(f"Bbox L1 (patch fraction): "
              f"{metrics['bbox_mae_patchfrac']:.4f}  "
              f"(over {metrics['bbox_n']} rows with target)")
    print("Per-class:")
    for cls, m in metrics['per_class'].items():
        print(f"  {cls:<9} p={m['precision']:.2f}  r={m['recall']:.2f}  "
              f"f1={m['f1']:.2f}   (support={m['support']})")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    cm = np.array(metrics['confusion_matrix'])
    print('         ' + ' '.join(f'{c[:4]:>5}' for c in metrics['class_labels']))
    for i, row in enumerate(cm):
        print(f"  {CLASS_LABELS[i][:8]:<8} " + ' '.join(f'{v:>5}' for v in row))
    print(f"\nCheckpoint: {save_path}")
    print(f"Metrics:    {metrics_path}")
    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True,
                        help='manifest_v2.csv from export_training_set_v2.')
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lambda_bbox', type=float, default=1.0,
                        help='Weight on the bbox-regression loss.')
    parser.add_argument('--patch_size', type=int, default=PATCH_DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train_profile_v2(**vars(args))


if __name__ == '__main__':
    main()
