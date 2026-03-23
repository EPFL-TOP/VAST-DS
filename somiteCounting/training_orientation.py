import os
import json
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from torchvision import models
from torchvision.models import ResNet18_Weights

from PIL import Image


# =========================================================
# PREPROCESSING (SHARED LOGIC)
# =========================================================
def preprocess_image(img_np, resize=(224, 224)):
    """
    Robust normalization + resize
    """
    img_np = img_np.astype(np.float32)

    # Robust normalization (avoid hot pixels dominating)
    p1, p99 = np.percentile(img_np, [1, 99])
    img_np = np.clip(img_np, p1, p99)
    img_np = (img_np - p1) / (p99 - p1 + 1e-6)

    # Convert to tensor
    #img = torch.from_numpy(img_np)[None, None]  # 1x1xH xW
    img = torch.from_numpy(img_np).float()[None, None]
    
    # Resize
    img = F.interpolate(img, size=resize, mode="bilinear", align_corners=False)

    return img.squeeze(0)  # 1xHxW


# =========================================================
# AUGMENTATION (SAFE FOR ORIENTATION TASK)
# =========================================================
class OrientationAugment:
    def __init__(self, resize=(224, 224), rotation=10,
                 brightness=0.2, contrast=0.2, noise=0.02):
        self.resize = resize
        self.rotation = rotation
        self.brightness = brightness
        self.contrast = contrast
        self.noise = noise

    def __call__(self, img_np):
        # --- preprocess first ---
        img = preprocess_image(img_np, self.resize)  # 1,H,W

        # Convert to PIL for geometric transforms
        #img_pil = Image.fromarray((img.squeeze().numpy() * 255).astype(np.uint8), mode="L")
        img_pil = Image.fromarray((img.squeeze().numpy() * 255).astype(np.uint8))
        # --- SMALL ROTATION (safe) ---
        if self.rotation > 0:
            angle = random.uniform(-self.rotation, self.rotation)
            img_pil = img_pil.rotate(angle)

        # Back to tensor
        #img = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0).unsqueeze(0)
        img = torch.from_numpy(np.array(img_pil)).float() / 255.0
        img = img.unsqueeze(0)

        # --- INTENSITY AUGMENT ---
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = img * factor

        if self.contrast > 0:
            mean = img.mean()
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = (img - mean) * factor + mean

        # --- NOISE ---
        if self.noise > 0:
            img = img + torch.randn_like(img) * self.noise

        img = torch.clamp(img, 0, 1)

        return img


# =========================================================
# DATASET
# =========================================================
class OrientationDataset(Dataset):
    """
    Each image.json contains {"correct_orientation": true/false}
    """

    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform

        self.samples = []
        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):

                if 'YFP' in fname:
                    continue

                img_path = os.path.join(folder, fname)
                json_path = os.path.splitext(img_path)[0] + ".json"

                if not os.path.exists(json_path):
                    continue

                with open(json_path, "r") as f:
                    info = json.load(f)

                if info.get("correct_orientation") is None:
                    continue
                if not info.get("valid", True):
                    continue

                self.samples.append((img_path, json_path))

        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # --- label ---
        with open(json_path, "r") as f:
            info = json.load(f)

        label = 1 if info.get("correct_orientation", False) else 0
        label = torch.tensor(label, dtype=torch.float32)

        # --- image ---
        img_np = np.array(Image.open(img_path)).astype(np.float32)

        if self.transform:
            img_tensor = self.transform(img_np)
        else:
            img_tensor = preprocess_image(img_np)

        return img_tensor, label


# =========================================================
# MODEL (PRETRAINED)
# =========================================================
class OrientationClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adapt first conv for grayscale
        old_conv1 = base.conv1
        base.conv1 = nn.Conv2d(
            1, old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False
        )

        base.conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)

        #base.fc = nn.Linear(base.fc.in_features, 1)

        # 👇 add dropout before FC
        base.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(base.fc.in_features, 1)
        )

        self.model = base

    def forward(self, x):
        return self.model(x)


# =========================================================
# TRAINING
# =========================================================
def train_orientation(data_folder,
                      save_path="orientation_best.pth",
                      epochs=30,
                      batch_size=8,
                      lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = OrientationAugment()

    train_dataset = OrientationDataset(
        os.path.join(data_folder, "train"),
        transform=transform
    )

    valid_dataset = OrientationDataset(
        os.path.join(data_folder, "valid"),
        transform=lambda x: preprocess_image(x)  # NO augmentation
    )

    print(f"Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = OrientationClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):

        # ================= TRAIN =================
        model.train()
        train_loss = 0
        correct = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()

        train_loss /= len(train_dataset)
        train_acc = correct / len(train_dataset)

        # ================= VALID =================
        model.eval()
        val_loss = 0
        correct = 0

        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).unsqueeze(1)

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()

        val_loss /= len(valid_dataset)
        val_acc = correct / len(valid_dataset)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict()
            }, save_path)
            print("  Saved best model.")

    print("Training finished.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    train_orientation(
        data_folder=r"D:\vast\VAST-DS\training_data",
        save_path=r"checkpoints\orientation_best.pth",
        epochs=40,
        batch_size=8,
        lr=1e-4
    )

