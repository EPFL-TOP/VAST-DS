# orientation_dataset.py
import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import torch.optim as optim

from PIL import Image


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
                    continue  # skip YFP images
                img_path = os.path.join(folder, fname)
                json_path = os.path.splitext(img_path)[0] + ".json"

                if not os.path.exists(json_path):
                    print(f"⚠ Missing JSON for {img_path}")
                    continue

                with open(json_path, "r") as file:
                    label_data = json.load(file)
                    if label_data["correct_orientation"] is None:
                        continue

                self.samples.append((img_path, json_path))

        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # Load label
        with open(json_path, "r") as f:
            info = json.load(f)
        label = 1 if info.get("correct_orientation", False) else 0
        label = torch.tensor(label, dtype=torch.float32)

        # Load grayscale image
        img_np = np.array(Image.open(img_path)).astype(np.float32)
        img_np /= img_np.max()  # normalize to 0-1

        if self.transform:
            img_tensor = self.transform(img_np)
        else:
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # 1,H,W

        return img_tensor, label




class OrientationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)  # train from scratch

        # Adapt for grayscale
        old_conv1 = base.conv1
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        nn.init.kaiming_normal_(base.conv1.weight, nonlinearity="relu")

        base.fc = nn.Linear(base.fc.in_features, 1)
        self.model = base

    def forward(self, x):
        return self.model(x)





# --- Transform ---
class GrayscaleTransform:
    def __init__(self, resize=(224, 224)):
        self.resize = resize

    def __call__(self, img_np):
        from torchvision.transforms import functional as TF
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8), mode="L")
        img_pil = img_pil.resize(self.resize)
        return TF.to_tensor(img_pil)



def train_orientation(data_folder, save_path="orientation_best.pth",
                      epochs=20, batch_size=8, lr=1e-4):

    # Dataset
    transform = GrayscaleTransform()
    dataset_train = OrientationDataset(os.path.join(data_folder,'train'), transform=transform)
    dataset_valid = OrientationDataset(os.path.join(data_folder, 'valid'), transform=transform)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset_valid,   batch_size=batch_size, shuffle=False)

    # Model
    model = OrientationClassifier().cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        running_corrects = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.cuda(), labels.cuda().unsqueeze(1)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            running_corrects += (preds == labels).sum().item()

        train_loss = running_loss / len(dataset_train)
        train_acc  = running_corrects / len(dataset_train)

        # Validation
        model.eval()
        val_loss = 0
        val_corrects = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.cuda(), labels.cuda().unsqueeze(1)

                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_corrects += (preds == labels).sum().item()

        val_loss /= len(dataset_valid)
        val_acc = val_corrects / len(dataset_valid)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, save_path)
            print("   Saved best model.")

    print("Training finished.")


# -----------------------------
if __name__ == "__main__":
    print("Training starts...")
    train_orientation(
        data_folder=r"D:\vast\training_data",
        save_path=r"checkpoints\orientation_best.pth",
        epochs=10,
        batch_size=16,
        lr=1e-4,
    )
    print("Training completed.")# orientation_dataset.py
