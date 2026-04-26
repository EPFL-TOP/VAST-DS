"""Somite-counting training script.

Predicts (n_total_somites, n_bad_somites) from a YFP fluorescence image of a
zebrafish. ResNet18-based regressor with weighted MSE loss; uses the shared
preprocessing + train loop from `_common`.

Public re-exports kept for backwards compatibility with code that imported
`SomiteCounter_freeze`, `FishQualityClassifier`, `WeightedMSELoss` directly
from this module (in particular, well_explorer/views.py).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn

# Allow running directly: `python somiteCounting/training.py`. When imported as
# part of the somiteCounting package (the normal case via Django), __package__
# is set and the sys.path tweak is a no-op.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from somiteCounting._common import (
    BaseAugment,
    ImageJsonDataset,
    WeightedMSELoss,
    evaluate_regression,
    make_grayscale_resnet18,
    preprocess_image,
    train_loop,
)


# =====================================================================
# Dataset
# =====================================================================
class SomiteDataset(ImageJsonDataset):
    """YFP images annotated with (n_total_somites, n_bad_somites, errors).

    Only `valid=True` samples are kept (training on invalid fish would be noise).
    """

    REQUIRED_KEYS = (
        "n_total_somites", "n_bad_somites",
        "n_total_somites_err", "n_bad_somites_err",
        "valid",
    )
    FILENAME_FILTER = "YFP"

    def accept_sample(self, info):
        return bool(info.get("valid", False))

    def extract_label(self, info):
        y = torch.tensor([info["n_total_somites"], info["n_bad_somites"]],
                         dtype=torch.float32)
        err = torch.tensor([info["n_total_somites_err"], info["n_bad_somites_err"]],
                           dtype=torch.float32)
        return (y, err)


# =====================================================================
# Augmentation policy: somite count is invariant under hflip but vflip would
# put dorsal where ventral is — keep that off.
# =====================================================================
class SomiteAugment(BaseAugment):
    def __init__(self, resize=(224, 224)):
        super().__init__(
            resize=resize,
            horizontal_flip=True,
            vertical_flip=False,
            rotation=10,
            brightness=0.15,
            contrast=0.15,
        )


# =====================================================================
# Models — thin wrappers over the shared factory, kept for backwards-compat
# =====================================================================
class SomiteCounter_freeze(nn.Module):
    """ResNet18 → 2 outputs (total, defective) with selective unfreezing."""

    def __init__(self, unfreeze_layers=("layer3", "layer4"), unfreeze_all=False):
        super().__init__()
        self.model = make_grayscale_resnet18(
            num_outputs=2,
            dropout=0.0,
            unfreeze_layers=unfreeze_layers,
            unfreeze_all=unfreeze_all,
        )

    def forward(self, x):
        return self.model(x)


class FishQualityClassifier(nn.Module):
    """ResNet18 → 1 logit (valid fish?). Re-exported here for back-compat.

    `dropout=0.0` by default to match the existing checkpoint's architecture.
    Pass `dropout=0.3` when training a new model if you want MC-dropout
    uncertainty (you will then also need to pass it when constructing the model
    for inference, or the state_dict will not load)."""

    def __init__(self, unfreeze_layers=("layer3", "layer4"), unfreeze_all=False,
                 dropout=0.0):
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
def train_somite_counter(input_data_path: str,
                         model_save_path: str,
                         epochs: int = 150,
                         batch_size: int = 8,
                         patience: int = 7,
                         lr_head: float = 1e-4,
                         lr_backbone: float = 1e-5):
    transform = SomiteAugment()
    train_dataset = SomiteDataset(os.path.join(input_data_path, "train"), transform=transform)
    valid_dataset = SomiteDataset(os.path.join(input_data_path, "valid"),
                                   transform=lambda x: preprocess_image(x))
    print(f"Somite training:   train={len(train_dataset)}, valid={len(valid_dataset)}")

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training samples in {input_data_path}/train")

    model = SomiteCounter_freeze()
    criterion = WeightedMSELoss()

    def _unpack(batch, device):
        img, y, err = batch
        return (img.to(device),), (y.to(device), err.to(device))

    save_file = os.path.join(model_save_path, "somite_counting_best.pth")
    model = train_loop(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        criterion=criterion,
        save_path=save_file,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr_head=lr_head,
        lr_backbone=lr_backbone,
        batch_unpacker=_unpack,
    )
    return model


def evaluate_test_set(model, input_data_path, model_save_path,
                      test_path: str = None, batch_size: int = 8):
    """Run the model on a held-out `test/` folder; print and persist metrics."""
    test_path = test_path or os.path.join(input_data_path, "test")
    if not os.path.isdir(test_path):
        print(f"[test eval] No test folder at {test_path} — skipping.")
        return None
    test_dataset = SomiteDataset(test_path, transform=lambda x: preprocess_image(x))
    if len(test_dataset) == 0:
        print(f"[test eval] Test folder {test_path} is empty — skipping.")
        return None
    print(f"[test eval] Held-out test set: {len(test_dataset)} images")
    report = evaluate_regression(model, test_dataset,
                                 output_names=("total", "bad"),
                                 batch_size=batch_size)
    for k, v in report.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    out_path = os.path.join(model_save_path, "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[test eval] Saved metrics to {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train somite counting model")
    parser.add_argument("--epochs", type=int, default=150)
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

    model = train_somite_counter(
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
