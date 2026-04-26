"""Fish-validity classifier training script.

Predicts whether a YFP image shows a valid fish (label = `valid` flag in
DestWellProperties). Binary classifier, ResNet18 + dropout. Uses the shared
preprocessing + train loop from `_common`.

Note: the production `FishQualityClassifier` class lives in `training.py` and
is imported by `well_explorer/views.py`. This script trains it.
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
    preprocess_image,
    train_loop,
)
from somiteCounting.training import FishQualityClassifier


# =====================================================================
# Dataset
# =====================================================================
class FishValidityDataset(ImageJsonDataset):
    """YFP images annotated with a `valid` boolean."""

    REQUIRED_KEYS = ("valid",)
    FILENAME_FILTER = "YFP"

    def extract_label(self, info):
        return torch.tensor(1.0 if info.get("valid", False) else 0.0,
                            dtype=torch.float32)


# =====================================================================
# Augmentation policy: validity is geometrically invariant — both flips OK.
# =====================================================================
class FishValidityAugment(BaseAugment):
    def __init__(self, resize=(224, 224)):
        super().__init__(
            resize=resize,
            horizontal_flip=True,
            vertical_flip=True,
            rotation=15,
            brightness=0.2,
            contrast=0.2,
        )


# =====================================================================
# Training entry point
# =====================================================================
def train_fish_validity(input_data_path: str,
                        model_save_path: str,
                        epochs: int = 50,
                        batch_size: int = 8,
                        patience: int = 7,
                        lr_head: float = 1e-4,
                        lr_backbone: float = 1e-5):
    transform = FishValidityAugment()
    train_dataset = FishValidityDataset(os.path.join(input_data_path, "train"),
                                        transform=transform)
    valid_dataset = FishValidityDataset(os.path.join(input_data_path, "valid"),
                                        transform=lambda x: preprocess_image(x))
    print(f"Validity training: train={len(train_dataset)}, valid={len(valid_dataset)}")

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training samples in {input_data_path}/train")

    model = FishQualityClassifier()

    bce = nn.BCEWithLogitsLoss()

    def _criterion(pred, label):
        # pred is shape [B, 1]; label is [B] — make shapes match
        return bce(pred.squeeze(-1), label)

    def _accuracy(pred, label):
        with torch.no_grad():
            p = (torch.sigmoid(pred.squeeze(-1)) > 0.5).float()
            return (p == label).float().mean().item()

    save_file = os.path.join(model_save_path, "fish_quality_best.pth")
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
    test_dataset = FishValidityDataset(test_path, transform=lambda x: preprocess_image(x))
    if len(test_dataset) == 0:
        print(f"[test eval] Test folder {test_path} is empty — skipping.")
        return None
    print(f"[test eval] Held-out test set: {len(test_dataset)} images")
    report = evaluate_classification(model, test_dataset, batch_size=batch_size)
    for k, v in report.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    out_path = os.path.join(model_save_path, "fish_quality_test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[test eval] Saved metrics to {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fish-validity classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--input_data_path", type=str, default=r"D:\vast\training_data")
    parser.add_argument("--model_save_path", type=str, default="checkpoints")
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--skip_test_eval", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.model_save_path, exist_ok=True)
    model = train_fish_validity(
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
