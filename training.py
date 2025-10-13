
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import pandas as pd
import os
import json


class SomiteDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        """
        img_dir: folder with images
        label_dir: folder with matching JSON files
        Assumes: image 'name.ext' has label 'name.json'
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # list all image files
        self.img_files = [f for f in os.listdir(img_dir) 
                          if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # load image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # load json label
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.label_dir, base_name + ".json")
        with open(label_path, "r") as f:
            label_data = json.load(f)

        # build target and error tensors
        y = torch.tensor([
            label_data["n_good_somites"], 
            label_data["n_bad_somites"]
        ], dtype=torch.float32)

        err = torch.tensor([
            label_data["n_good_somites_err"], 
            label_data["n_bad_somites_err"]
        ], dtype=torch.float32)

        return img, y, err
    


# -----------------------------
# Model
# -----------------------------
class SomiteCounter(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Linear(base.fc.in_features, 2)  # two outputs
        self.model = base

    def forward(self, x):
        return self.model(x)


# -----------------------------
# Weighted loss function
# -----------------------------
class WeightedMSELoss(nn.Module):
    def forward(self, pred, target, error):
        # avoid division by zero
        error = torch.clamp(error, min=1.0)  
        return torch.mean(((pred - target) ** 2) / (error ** 2))



# -----------------------------
# Training loop
# -----------------------------
def train_model(train_dataset, valid_dataset, 
                save_dir="checkpoints",
                epochs=50, batch_size=8, lr=1e-4, patience=5,
                resume=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = SomiteCounter().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = WeightedMSELoss()

    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = 0
    epochs_no_improve = 0

    # --- Resume training if requested ---
    if resume:
        checkpoint_path = os.path.join(save_dir, "best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["val_loss"]
            print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}")

    # --- Main loop ---
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        for imgs, labels, errors in train_loader:
            imgs, labels, errors = imgs.to(device), labels.to(device), errors.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels, errors)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels, errors in valid_loader:
                imgs, labels, errors = imgs.to(device), labels.to(device), errors.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels, errors)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(valid_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, os.path.join(save_dir, "best_model.pth"))
            print("Saved best model")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    return model




transform = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
])

train_dataset = SomiteDataset(r"D:\vast\training_data\train", r"D:\vast\training_data\train", transform=transform)
valid_dataset = SomiteDataset(r"D:\vast\training_data\valid", r"D:\vast\training_data\valid", transform=transform)

model = train_model(train_dataset, valid_dataset, save_dir="checkpoints", epochs=50, patience=7)



checkpoint = torch.load("checkpoints/best_model.pth", map_location="cpu")
model = SomiteCounter()
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1

