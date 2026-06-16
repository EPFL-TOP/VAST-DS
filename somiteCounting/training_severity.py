"""Severity classifier — train a 4-way CNN on hand-annotated somite tiles.

Reads ``manifest.csv`` produced by ``export_somite_training_set``,
trains a ResNet18 (ImageNet-pretrained, 1-channel input) on the
severity label (0 healthy / 1 mild / 2 moderate / 3 severe), and saves:

  * ``checkpoints/severity_best.pth``         — model weights
  * ``checkpoints/severity_test_metrics.json`` — per-class precision/recall/
    F1 + confusion matrix on the held-out test fish

Splits **by fish** (``dest_well_id``), not by somite: somites from the
same well stay in the same split, so val/test scores aren't inflated by
the model seeing nearby somites of the same fish in training.

Class imbalance: weighted ``CrossEntropyLoss`` with weights = 1/sqrt(freq)
(sqrt-scaled — pure 1/freq is too aggressive when one class has <10
samples, the rare-class loss dominates and accuracy collapses).

Augmentation: horizontal flip only (along AP axis). Bilateral symmetry
makes left/right flip safe; rotation isn't safe because head/tail
orientation is semantically meaningful.

Examples
--------
    python -m somiteCounting.training_severity \
        --manifest data/somite_training_set/manifest.csv

    python -m somiteCounting.training_severity \
        --manifest data/training_v1/manifest.csv \
        --output_dir checkpoints --epochs 80 --batch_size 16
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Allow `python somiteCounting/training_severity.py` AND
# `python -m somiteCounting.training_severity`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from somiteCounting._common import (  # noqa: E402
    _device, make_grayscale_resnet18, preprocess_image, train_loop,
)


SEVERITY_LABELS = ['healthy', 'mild', 'moderate', 'severe']
NUM_CLASSES = 4


# =====================================================================
# Dataset
# =====================================================================
class SeverityDataset(Dataset):
    """Tile + label pairs from a flat list of ``(path, severity)`` tuples.

    The well-grouped split is done by the caller, so this class only
    consumes whichever samples are passed in. Augmentation is a single
    coin-flip horizontal mirror — safe under the fish's bilateral
    symmetry. We deliberately don't rotate (orientation matters).
    """

    def __init__(self, samples, augment: bool = False):
        self.samples = list(samples)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, sev = self.samples[idx]
        img_np = np.array(Image.open(path)).astype(np.float32)
        if self.augment and np.random.rand() < 0.5:
            img_np = np.ascontiguousarray(np.fliplr(img_np))
        img = preprocess_image(img_np)  # (1, H, W) float tensor
        return img, torch.tensor(sev, dtype=torch.long)


# =====================================================================
# Manifest I/O + splitting
# =====================================================================
def _read_manifest(manifest_path: str):
    """Return a list of dicts from manifest.csv. Coerces numeric fields."""
    rows = []
    with open(manifest_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['severity'] = int(row['severity'])
                row['dest_well_id'] = int(row['dest_well_id'])
            except (KeyError, ValueError):
                continue  # malformed row — skip silently
            rows.append(row)
    return rows


def _split_by_well(rows, *, train_pct=0.70, val_pct=0.15, seed=42):
    """Group by ``dest_well_id`` then split the *wells* into train/val/test.

    Returns three lists of rows. Same fish never appears in two splits.
    """
    by_well = defaultdict(list)
    for r in rows:
        by_well[r['dest_well_id']].append(r)
    wells = sorted(by_well.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(wells)
    n = len(wells)
    n_train = int(round(n * train_pct))
    n_val = int(round(n * val_pct))
    train_wells = wells[:n_train]
    val_wells = wells[n_train:n_train + n_val]
    test_wells = wells[n_train + n_val:]
    train = [r for w in train_wells for r in by_well[w]]
    val = [r for w in val_wells for r in by_well[w]]
    test = [r for w in test_wells for r in by_well[w]]
    return train, val, test


# =====================================================================
# Class weighting
# =====================================================================
def _class_weights_sqrt_inv(samples, num_classes: int = NUM_CLASSES):
    """1/sqrt(freq), normalised so the mean weight is 1.

    Pure 1/freq blows up when one class has <10 samples (one mild somite
    in 1000 means the mild loss is 1000× a healthy somite, and the model
    learns to predict 'mild' for everything). sqrt-scaling tames that.
    Empty classes get freq=1 so they don't divide by zero (the resulting
    weight is harmless — no samples to apply it to).
    """
    cnt = Counter(s[1] for s in samples)
    freqs = np.array([max(cnt.get(c, 0), 1) for c in range(num_classes)],
                     dtype=np.float64)
    w = 1.0 / np.sqrt(freqs)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# =====================================================================
# Evaluation (multi-class)
# =====================================================================
def _evaluate_multiclass(model, dataset, *,
                         num_classes: int = NUM_CLASSES,
                         batch_size: int = 16):
    """Per-class precision/recall/F1 + confusion matrix + accuracy."""
    device = _device()
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(device)
            preds = model(img).argmax(dim=1).cpu().numpy()
            true = lbl.numpy()
            for t, p in zip(true, preds):
                cm[t, p] += 1

    per_class = {}
    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum()) - tp
        fp = int(cm[:, c].sum()) - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        per_class[SEVERITY_LABELS[c]] = {
            'precision': float(p), 'recall': float(r), 'f1': float(f1),
            'support': int(cm[c, :].sum()),
        }
    n = int(cm.sum())
    acc = float(cm.trace()) / max(n, 1)
    return {
        'n': n,
        'accuracy': acc,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'class_labels': SEVERITY_LABELS,
    }


# =====================================================================
# Top-level orchestration
# =====================================================================
def train_severity(*,
                   manifest: str,
                   output_dir: str = 'checkpoints',
                   epochs: int = 50,
                   batch_size: int = 16,
                   patience: int = 8,
                   lr_head: float = 1e-4,
                   lr_backbone: float = 1e-5,
                   dropout: float = 0.3,
                   seed: int = 42):
    rows = _read_manifest(manifest)
    if not rows:
        raise SystemExit(f"manifest is empty / unreadable: {manifest}")
    print(f"Read {len(rows)} rows from {manifest}")

    train_rows, val_rows, test_rows = _split_by_well(rows, seed=seed)
    print(f"Split (by fish): train={len(train_rows)} "
          f"val={len(val_rows)} test={len(test_rows)} "
          f"(across {len({r['dest_well_id'] for r in rows})} wells)")

    # Build (path, label) tuples — manifest tile_path is relative to the
    # manifest's parent directory (the export's --output_dir).
    manifest_root = os.path.dirname(os.path.abspath(manifest))
    def _resolve(r):
        return (os.path.join(manifest_root, r['tile_path']), r['severity'])

    train_samples = [_resolve(r) for r in train_rows]
    val_samples   = [_resolve(r) for r in val_rows]
    test_samples  = [_resolve(r) for r in test_rows]

    train_freqs = Counter(s[1] for s in train_samples)
    print(f"Train class distribution: "
          f"{ {SEVERITY_LABELS[c]: train_freqs.get(c, 0) for c in range(NUM_CLASSES)} }")
    if any(train_freqs.get(c, 0) < 5 for c in range(NUM_CLASSES)):
        print("  WARNING: at least one class has <5 train samples — "
              "results will be unreliable. Annotate more and re-run.")

    class_weights = _class_weights_sqrt_inv(train_samples)
    print(f"Class weights (1/sqrt(freq), mean-normalised): "
          f"{ {SEVERITY_LABELS[c]: round(float(class_weights[c]), 3) for c in range(NUM_CLASSES)} }")

    train_ds = SeverityDataset(train_samples, augment=True)
    val_ds   = SeverityDataset(val_samples,   augment=False)
    test_ds  = SeverityDataset(test_samples,  augment=False)

    device = _device()
    print(f"Using device: {device}")
    model = make_grayscale_resnet18(num_outputs=NUM_CLASSES, dropout=dropout)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'severity_best.pth')

    model = train_loop(
        model, train_ds, val_ds, criterion,
        save_path=save_path,
        epochs=epochs, batch_size=batch_size,
        lr_head=lr_head, lr_backbone=lr_backbone,
        patience=patience,
    )

    # Held-out test evaluation
    print("\nEvaluating on test set…")
    if not test_samples:
        print("  (no test samples — too few wells; metrics file not written)")
        return model, None

    metrics = _evaluate_multiclass(model, test_ds, batch_size=batch_size)
    metrics_path = os.path.join(output_dir, 'severity_test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTest set: N={metrics['n']}  accuracy={metrics['accuracy']:.3f}")
    print("Per-class:")
    for cls, m in metrics['per_class'].items():
        print(f"  {cls:<9} p={m['precision']:.2f}  r={m['recall']:.2f}  "
              f"f1={m['f1']:.2f}   (support={m['support']})")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    cm = np.array(metrics['confusion_matrix'])
    header = '         ' + ' '.join(f'{c[:4]:>5}' for c in metrics['class_labels'])
    print(header)
    for i, row in enumerate(cm):
        print(f"  {SEVERITY_LABELS[i][:8]:<8} " +
              ' '.join(f'{v:>5}' for v in row))
    print(f"\nCheckpoint: {save_path}")
    print(f"Metrics:    {metrics_path}")
    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True,
                        help='manifest.csv from export_somite_training_set.')
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train_severity(**vars(args))


if __name__ == '__main__':
    main()
