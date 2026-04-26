"""Shared building blocks for the three training scripts in this package
(somite counting, fish validity, orientation).

Public surface:
- preprocess_image(): the ONE normalisation function — percentile clip + resize.
  Use it at training, evaluation, and inference. Same function everywhere.
- ImageJsonDataset: base class for "folder of images + matching .json sidecars".
- BaseAugment: configurable geometric + intensity augmentation. Calls
  preprocess_image first so augmented samples share the same normalisation.
- make_grayscale_resnet18(): factory for ResNet18 adapted to 1-channel input
  with optional dropout and configurable layer freezing.
- WeightedMSELoss: regression loss that down-weights uncertain labels.
- train_loop(): generic train+validate loop with early stopping and best-model
  checkpointing. Works for both classification (BCE) and regression (MSE).
- evaluate_regression() / evaluate_classification(): metric reports for the
  held-out test set.
"""

import os
import json
import random
from typing import Callable, Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights
import torchvision.transforms.functional as TF


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# =====================================================================
# Preprocessing — used everywhere (training, evaluation, dashboard)
# =====================================================================
def preprocess_image(img_np: np.ndarray, resize=(224, 224)) -> torch.Tensor:
    """Normalise a 2-D grayscale image and resize it.

    Uses 1st/99th percentile clipping (robust to hot pixels common in YFP
    fluorescence). Returns a (1, H, W) float tensor in [0, 1].
    """
    img_np = img_np.astype(np.float32)
    p1, p99 = np.percentile(img_np, [1, 99])
    img_np = np.clip(img_np, p1, p99)
    img_np = (img_np - p1) / (p99 - p1 + 1e-6)

    img = torch.from_numpy(img_np).float()[None, None]   # 1,1,H,W
    img = F.interpolate(img, size=resize, mode="bilinear", align_corners=False)
    return img.squeeze(0)   # 1,H,W


# =====================================================================
# Augmentation
# =====================================================================
class BaseAugment:
    """Configurable augmentation. Always normalises via preprocess_image first.

    Per-task policy:
    - somite counting: hflip OK (count is invariant), no vflip.
    - validity:        hflip + vflip OK.
    - orientation:     no flips, small rotation only — flips would invert the label.
    """

    def __init__(self,
                 resize: Tuple[int, int] = (224, 224),
                 horizontal_flip: bool = False,
                 vertical_flip: bool = False,
                 rotation: float = 0.0,
                 brightness: float = 0.0,
                 contrast: float = 0.0,
                 noise: float = 0.0):
        self.resize = resize
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.brightness = brightness
        self.contrast = contrast
        self.noise = noise

    def __call__(self, img_np: np.ndarray) -> torch.Tensor:
        # 1) normalise + resize (same code path as inference)
        img = preprocess_image(img_np, self.resize)         # 1,H,W float [0,1]

        # 2) geometric (use PIL via TF helpers — works on 1-channel tensors)
        if self.horizontal_flip and random.random() < 0.5:
            img = TF.hflip(img)
        if self.vertical_flip and random.random() < 0.5:
            img = TF.vflip(img)
        if self.rotation > 0:
            angle = random.uniform(-self.rotation, self.rotation)
            img = TF.rotate(img, angle, fill=0.0)

        # 3) intensity
        if self.brightness > 0:
            img = img * (1.0 + random.uniform(-self.brightness, self.brightness))
        if self.contrast > 0:
            mean = img.mean()
            img = (img - mean) * (1.0 + random.uniform(-self.contrast, self.contrast)) + mean
        if self.noise > 0:
            img = img + torch.randn_like(img) * self.noise

        return torch.clamp(img, 0.0, 1.0)


# =====================================================================
# Dataset base
# =====================================================================
class ImageJsonDataset(Dataset):
    """Walk a folder; pair every image with its .json sidecar; expose a
    PyTorch Dataset.

    Subclasses customise:
        REQUIRED_KEYS:   tuple of keys that must exist in the JSON
        FILENAME_FILTER: 'YFP' to keep only YFP files, 'NOT_YFP' to drop them,
                         or None to keep all
        accept_sample(self, info) -> bool: filter rows on label content
        extract_label(self, info) -> tensor or tuple of tensors

    Each __getitem__ returns (image_tensor, *label_tensors) so subclasses
    that need an error tensor can return one.
    """

    REQUIRED_KEYS: Tuple[str, ...] = ()
    FILENAME_FILTER: Optional[str] = None   # 'YFP', 'NOT_YFP', or None

    def __init__(self, folder: str, transform: Optional[Callable] = None):
        self.folder = folder
        self.transform = transform
        self.samples: List[Tuple[str, str]] = []

        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            if self.FILENAME_FILTER == 'YFP' and 'YFP' not in fname:
                continue
            if self.FILENAME_FILTER == 'NOT_YFP' and 'YFP' in fname:
                continue

            img_path = os.path.join(folder, fname)
            json_path = os.path.splitext(img_path)[0] + ".json"
            if not os.path.exists(json_path):
                continue

            with open(json_path, "r") as f:
                info = json.load(f)

            if self.REQUIRED_KEYS and not all(k in info for k in self.REQUIRED_KEYS):
                continue
            if not self.accept_sample(info):
                continue

            self.samples.append((img_path, json_path))

    # --- to be overridden by subclasses ---
    def accept_sample(self, info: dict) -> bool:
        return True

    def extract_label(self, info: dict):
        """Return a tensor or a tuple of tensors representing the label(s)."""
        raise NotImplementedError

    # --- common ---
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        img_np = np.array(Image.open(img_path)).astype(np.float32)
        if self.transform is not None:
            img = self.transform(img_np)
        else:
            img = preprocess_image(img_np)
        with open(json_path, "r") as f:
            info = json.load(f)
        label = self.extract_label(info)
        if isinstance(label, tuple):
            return (img, *label)
        return img, label


# =====================================================================
# Model factory
# =====================================================================
def make_grayscale_resnet18(num_outputs: int,
                            dropout: float = 0.0,
                            unfreeze_layers: Tuple[str, ...] = ("layer3", "layer4"),
                            unfreeze_all: bool = False) -> nn.Module:
    """Return a ResNet18 adapted to 1-channel input with `num_outputs` head.

    The pretrained `conv1` weights are averaged across the RGB channels so
    the network keeps useful low-level filters. If `dropout > 0`, a Dropout
    is inserted before the final linear layer.
    """
    base = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    old_conv1 = base.conv1
    base.conv1 = nn.Conv2d(
        1, old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=False,
    )
    base.conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)

    in_features = base.fc.in_features
    if dropout > 0:
        base.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_outputs))
    else:
        base.fc = nn.Linear(in_features, num_outputs)

    # Freezing policy
    if not unfreeze_all:
        for p in base.parameters():
            p.requires_grad = False
        # always train head + adapted conv1
        for p in base.fc.parameters():
            p.requires_grad = True
        for p in base.conv1.parameters():
            p.requires_grad = True
        for name, module in base.named_children():
            if name in unfreeze_layers:
                for p in module.parameters():
                    p.requires_grad = True

    return base


# =====================================================================
# Loss
# =====================================================================
class WeightedMSELoss(nn.Module):
    """MSE divided by per-sample annotation uncertainty squared.

    Annotators flag how confident they are; this loss down-weights uncertain
    labels. Errors are clamped to >=1 so a "perfect" label doesn't blow up
    the loss for that sample.
    """

    def forward(self, pred, target, error):
        error = torch.clamp(error, min=1.0)
        return torch.mean(((pred - target) ** 2) / (error ** 2))


# =====================================================================
# Train loop — generic
# =====================================================================
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(model: nn.Module,
               train_dataset: Dataset,
               valid_dataset: Dataset,
               criterion,
               *,
               save_path: str,
               epochs: int = 50,
               batch_size: int = 8,
               lr_head: float = 1e-4,
               lr_backbone: float = 1e-5,
               patience: int = 7,
               batch_unpacker: Optional[Callable] = None,
               extra_metric: Optional[Callable] = None,
               verbose: bool = True) -> nn.Module:
    """Train `model` and save the best checkpoint by validation loss.

    `batch_unpacker(batch, device) -> (model_inputs, criterion_args)` lets the
    caller adapt to datasets that yield extra tensors (e.g. somite errors).
    Default unpacker assumes batches are (img, label).
    """
    device = _device()
    model = model.to(device)

    # Differential learning rates: head + adapted conv1 fast, rest slow.
    # We identify head parameters by name (`.fc.` or `.conv1.` anywhere in the
    # dotted path) so the wrapper classes don't have to expose those submodules
    # as direct attributes — that would change the state_dict keys and break
    # loading of existing checkpoints.
    HEAD_SEGMENTS = {"fc", "conv1"}
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(seg in HEAD_SEGMENTS for seg in name.split(".")):
            head_params.append(p)
        else:
            backbone_params.append(p)
    optimizer = optim.Adam([
        {"params": head_params,     "lr": lr_head},
        {"params": backbone_params, "lr": lr_backbone},
    ])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    if batch_unpacker is None:
        def batch_unpacker(batch, device):
            img, label = batch
            return (img.to(device),), (label.to(device),)

    best_val = float("inf")
    no_improve = 0

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ----- train -----
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in train_loader:
            (model_inputs, crit_args) = batch_unpacker(batch, device)
            preds = model(*model_inputs)
            loss = criterion(preds, *crit_args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = model_inputs[0].size(0)
            train_loss_sum += loss.item() * n
            train_n += n
        train_loss = train_loss_sum / max(train_n, 1)

        # ----- valid -----
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        extra_acc = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                (model_inputs, crit_args) = batch_unpacker(batch, device)
                preds = model(*model_inputs)
                loss = criterion(preds, *crit_args)
                n = model_inputs[0].size(0)
                val_loss_sum += loss.item() * n
                val_n += n
                if extra_metric is not None:
                    extra_acc += extra_metric(preds, *crit_args) * n
        val_loss = val_loss_sum / max(val_n, 1)

        msg = f"Epoch {epoch:3d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f}"
        if extra_metric is not None:
            msg += f" | extra {extra_acc/max(val_n,1):.4f}"

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": val_loss}, save_path)
            msg += "  ← best, saved"
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(msg)
                    print(f"Early stopping after {patience} epochs without improvement.")
                break

        if verbose:
            print(msg)

    # Reload best
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# =====================================================================
# Evaluation
# =====================================================================
def _inference_pass(model, dataset, batch_size):
    device = _device()
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch[0].to(device)
            label = batch[1]
            out = model(img).cpu().numpy()
            preds_all.append(out)
            labels_all.append(np.asarray(label))
    if not preds_all:
        return np.zeros((0, 1)), np.zeros((0,))
    return np.concatenate(preds_all, axis=0), np.concatenate(labels_all, axis=0)


def evaluate_regression(model,
                        dataset,
                        output_names: Tuple[str, ...] = ("total", "bad"),
                        batch_size: int = 8) -> Dict[str, Any]:
    """MAE / RMSE / signed-bias / R² per output, raw and integer-clamped."""
    preds, labels = _inference_pass(model, dataset, batch_size)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    def _metrics(p, t):
        if len(t) == 0:
            return {"n": 0}
        diff = p - t
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum()) if t.var() > 0 else 0.0
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {"n": int(len(t)),
                "mae": float(np.abs(diff).mean()),
                "rmse": float(np.sqrt((diff ** 2).mean())),
                "bias": float(diff.mean()),
                "r2": float(r2)}

    out: Dict[str, Any] = {}
    for i, name in enumerate(output_names):
        p, t = preds[:, i].astype(np.float64), labels[:, i].astype(np.float64)
        for k, v in _metrics(p, t).items():
            out[f"{name}_raw_{k}"] = v
        pi = np.clip(np.round(p), 0, None)
        for k, v in _metrics(pi, t).items():
            out[f"{name}_int_{k}"] = v
    return out


def evaluate_classification(model, dataset, batch_size: int = 8,
                            threshold: float = 0.5) -> Dict[str, Any]:
    """Accuracy / precision / recall / F1 / confusion-matrix for binary labels."""
    preds, labels = _inference_pass(model, dataset, batch_size)
    probs = 1.0 / (1.0 + np.exp(-preds.flatten()))
    pred_lbl = (probs > threshold).astype(int)
    true_lbl = labels.flatten().astype(int)
    if len(true_lbl) == 0:
        return {"n": 0}
    tp = int(((pred_lbl == 1) & (true_lbl == 1)).sum())
    tn = int(((pred_lbl == 0) & (true_lbl == 0)).sum())
    fp = int(((pred_lbl == 1) & (true_lbl == 0)).sum())
    fn = int(((pred_lbl == 0) & (true_lbl == 1)).sum())
    n = len(true_lbl)
    acc = (tp + tn) / n
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"n": n, "accuracy": acc, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
