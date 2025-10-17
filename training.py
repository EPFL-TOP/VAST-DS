
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
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights

# -----------------------------
# Dataset for 16-bit grayscale images
# -----------------------------
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
        self.img_files = [f for f in os.listdir(img_dir) 
                          if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load grayscale 16-bit image and scale to [0,1]
        img = np.array(Image.open(img_path)).astype(np.float32) / 65535.0

        if self.transform:
            img = self.transform(img)

        # load json label
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.label_dir, base_name + ".json")
        with open(label_path, "r") as f:
            label_data = json.load(f)

        # build target and error tensors
        y = torch.tensor([
            label_data["n_total_somites"], 
            label_data["n_bad_somites"]
        ], dtype=torch.float32)

        err = torch.tensor([
            label_data["n_total_somites_err"], 
            label_data["n_bad_somites_err"]
        ], dtype=torch.float32)

        return img, y, err


import torchvision.transforms.functional as TF

class GrayscaleTransform:
    """Apply augmentations to a 16-bit grayscale image and convert to tensor"""
    def __init__(self, resize=(224,224), horizontal_flip=True, rotation=10):
        self.resize = resize
        self.horizontal_flip = horizontal_flip
        self.rotation = rotation

    def __call__(self, img_np):
        """
        img_np: numpy array HxW, dtype uint16
        """
        # Convert to PIL (mode 'I;16') to preserve 16-bit info
        img_pil = Image.fromarray(img_np)

        # Resize
        img_pil = img_pil.resize(self.resize, resample=Image.BILINEAR)

        # Random horizontal flip
        if self.horizontal_flip and np.random.rand() > 0.5:
            img_pil = TF.hflip(img_pil)

        # Random rotation
        if self.rotation > 0:
            angle = np.random.uniform(-self.rotation, self.rotation)
            img_pil = TF.rotate(img_pil, angle)

        # Convert to numpy float32
        img_np = np.array(img_pil).astype(np.float32)

        # Scale to [0,1]
        img_np /= 65535.0

        # Convert to tensor
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # 1,H,W

        return img_tensor


# -----------------------------
# Custom transform for 16-bit grayscale to tensor
# -----------------------------
class ToTensorGrayscale:
    def __call__(self, img):
        # img is numpy HxW in [0,1]
        return torch.from_numpy(img).unsqueeze(0)  # 1,H,W


# -----------------------------
# Model for 1-channel input
# -----------------------------
class SomiteCounter(nn.Module):
    def __init__(self):
        super().__init__()
        #base = models.resnet18(pretrained=False)
        base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Linear(base.fc.in_features, 2)
        self.model = base

    def forward(self, x):
        return self.model(x)


# -----------------------------
# Weighted loss function
# -----------------------------
class WeightedMSELoss(nn.Module):
    def forward(self, pred, target, error):
        error = torch.clamp(error, min=1.0)
        return torch.mean(((pred - target) ** 2) / (error ** 2))


# -----------------------------
# Visualization helper
# -----------------------------
def show_image_comparison(raw_img, tensor_img):
    """
    raw_img: numpy HxW in [0,1]
    tensor_img: torch tensor 1xHxW
    """
    tensor_img = tensor_img.detach().cpu().squeeze(0).numpy()
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    axes[0].imshow(raw_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original 16-bit scaled")
    axes[0].axis("off")
    axes[1].imshow(tensor_img, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Network Input")
    axes[1].axis("off")
    plt.show()

# -----------------------------
# Training loop
# -----------------------------
def train_model_old(train_dataset, valid_dataset, 
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



    


# -----------------------------
# Training loop
# -----------------------------
def train_model(train_dataset, valid_dataset,
                save_dir="checkpoints",
                epochs=50, batch_size=8, lr=1e-4, patience=5,
                resume=False, visualize_every=5):

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

    # Resume
    if resume:
        checkpoint_path = os.path.join(save_dir, "best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["val_loss"]
            print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}")

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
            print(" Saved best model")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

        # Visualize one training image every few epochs
        if epoch % visualize_every == 0:
            sample_img, _, _ = train_dataset[0]
            show_image_comparison(sample_img.numpy().squeeze(), sample_img)

    return model





if __name__ == "__main__":
    print("Training starts...")

    # ------------------------
    # Transforms for 16-bit grayscale
    # ------------------------
    #transform = T.Compose([
    #    ToTensorGrayscale(),  # converts HxW numpy to 1xHxW tensor
    #    T.Resize((224,224)),  # resize to network input
    #    T.RandomHorizontalFlip(),
    #    T.RandomRotation(10),
    #    T.ColorJitter(brightness=0.2, contrast=0.2),  # optional augmentation
    #])

    transform = GrayscaleTransform(resize=(224,224), horizontal_flip=True, rotation=10)


    # ------------------------
    # Datasets
    # ------------------------
    train_dataset = SomiteDataset(
        img_dir=r"D:\vast\training_data\train",
        label_dir=r"D:\vast\training_data\train",
        transform=transform
    )

    valid_dataset = SomiteDataset(
        img_dir=r"D:\vast\training_data\valid",
        label_dir=r"D:\vast\training_data\valid",
        transform=transform
    )

    # ------------------------
    # Train the model
    # ------------------------
    model = train_model(
        train_dataset,
        valid_dataset,
        save_dir="checkpoints",
        epochs=50,
        batch_size=8,
        lr=1e-4,
        patience=7,
        resume=True,        # resume if checkpoint exists
        visualize_every=5   # show a sample image every 5 epochs
    )

    # ------------------------
    # Load the best checkpoint after training
    # ------------------------
    checkpoint_path = "checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = SomiteCounter()
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        print(f"Best model loaded from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
    else:
        print("No checkpoint found, using trained model as-is.")
