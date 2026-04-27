"""Orientation-classifier training script.

Predicts the `correct_orientation` flag in DestWellProperties (head should be
on the left side of the BF image). Trained on **brightfield** images (filenames
*without* 'YFP' — orientation is more visible there than in YFP).

Re-exports `preprocess_image` and `OrientationClassifier` for back-compat with
`well_explorer/views.py` and `orientfish.py`.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from somiteCounting._common import (
    BaseAugment,
    ImageJsonDataset,
    evaluate_classification,
    make_grayscale_resnet18,
    preprocess_image,   # re-exported for back-compat (well_explorer/views.py, orientfish.py)
    train_loop,
)


# =====================================================================
# Dataset — BF images annotated with `correct_orientation`
# =====================================================================
class OrientationDataset(ImageJsonDataset):
    REQUIRED_KEYS = ()                 # we accept missing keys, then filter below
    FILENAME_FILTER = "NOT_YFP"        # orientation labels live on BF images

    def accept_sample(self, info):
        if info.get("correct_orientation") is None:
            return False
        if not info.get("valid", True):
            return False
        return True

    def extract_label(self, info):
        return torch.tensor(1.0 if info.get("correct_orientation", False) else 0.0,
                            dtype=torch.float32)


# =====================================================================
# Augmentation policy: NEVER flip — flipping inverts the label.
# Small rotation only, plus intensity jitter and a touch of noise.
# =====================================================================
class OrientationAugment(BaseAugment):
    def __init__(self, resize=(224, 224)):
        super().__init__(
            resize=resize,
            horizontal_flip=False,
            vertical_flip=False,
            rotation=10,
            brightness=0.2,
            contrast=0.2,
            noise=0.02,
        )


# =====================================================================
# Model — kept as a class for back-compat; well_explorer/views.py imports it.
# =====================================================================
class OrientationClassifier(nn.Module):
    def __init__(self, dropout=0.3,
                 unfreeze_layers=("layer3", "layer4"), unfreeze_all=False):
        super().__init__()
        self.model = make_grayscale_resnet18(
            num_outputs=1,
            dropout=dropout,
            unfreeze_layers=unfreeze_layers,
            unfreeze_all=unfreeze_all,
        )

    def forward(self, x):
        return self.model(x)


# =====================================================================
# Training entry point
# =====================================================================
def train_orientation(input_data_path: str,
                      model_save_path: str,
                      epochs: int = 40,
                      batch_size: int = 8,
                      patience: int = 7,
                      lr_head: float = 1e-4,
                      lr_backbone: float = 1e-5):
    transform = OrientationAugment()
    train_dataset = OrientationDataset(os.path.join(input_data_path, "train"),
                                       transform=transform)
    valid_dataset = OrientationDataset(os.path.join(input_data_path, "valid"),
                                       transform=lambda x: preprocess_image(x))
    print(f"Orientation training: train={len(train_dataset)}, valid={len(valid_dataset)}")

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training samples in {input_data_path}/train")

    model = OrientationClassifier()

    bce = nn.BCEWithLogitsLoss()

    def _criterion(pred, label):
        return bce(pred.squeeze(-1), label)

    def _accuracy(pred, label):
        with torch.no_grad():
            p = (torch.sigmoid(pred.squeeze(-1)) > 0.5).float()
            return (p == label).float().mean().item()

    save_file = os.path.join(model_save_path, "orientation_best.pth")
    model = train_loop(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        criterion=_criterion,
        save_path=save_file,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr_head=lr_head,
        lr_backbone=lr_backbone,
        extra_metric=_accuracy,
    )
    return model


def evaluate_test_set(model, input_data_path, model_save_path,
                      test_path: str = None, batch_size: int = 8):
    test_path = test_path or os.path.join(input_data_path, "test")
    if not os.path.isdir(test_path):
        print(f"[test eval] No test folder at {test_path} — skipping.")
        return None
    test_dataset = OrientationDataset(test_path, transform=lambda x: preprocess_image(x))
    if len(test_dataset) == 0:
        print(f"[test eval] Test folder {test_path} is empty — skipping.")
        return None
    print(f"[test eval] Held-out test set: {len(test_dataset)} images")
    report = evaluate_classification(model, test_dataset, batch_size=batch_size)
    for k, v in report.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    out_path = os.path.join(model_save_path, "orientation_test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[test eval] Saved metrics to {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train orientation classifier")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--input_data_path", type=str, default=r"D:\vast\VAST-DS\training_data")
    parser.add_argument("--model_save_path", type=str, default="checkpoints")
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--skip_test_eval", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.model_save_path, exist_ok=True)
    model = train_orientation(
        input_data_path=args.input_data_path,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
    )
    if not args.skip_test_eval:
        evaluate_test_set(model, args.input_data_path, args.model_save_path,
                          test_path=args.test_data_path,
                          batch_size=args.batch_size)
