
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
import torchvision.transforms.functional as TF
import random

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
        img_np = np.array(Image.open(img_path)).astype(np.float32)
        img_np /= img_np.max()  # normalize to 0-1

        if self.transform:
            img_tensor = self.transform(img_np)
        else:
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # 1,H,W

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

        return img_tensor, y, err



# -----------------------------
# Transform / augmentation for grayscale
# -----------------------------
class GrayscaleAugment:
    def __init__(self, resize=(224,224), horizontal_flip=True, rotation=10):
        self.resize = resize
        self.horizontal_flip = horizontal_flip
        self.rotation = rotation

    def __call__(self, img_np):
        # Convert to PIL for augmentations
        img_pil = Image.fromarray((img_np*65535).astype(np.uint16))

        # Resize
        img_pil = img_pil.resize(self.resize, resample=Image.BILINEAR)

        # Horizontal flip
        if self.horizontal_flip and np.random.rand() > 0.5:
            img_pil = TF.hflip(img_pil)

        # Random rotation
        if self.rotation > 0:
            angle = np.random.uniform(-self.rotation, self.rotation)
            img_pil = TF.rotate(img_pil, angle)

        # Convert back to float32 0-1
        img_np = np.array(img_pil).astype(np.float32) / 65535.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # 1,H,W
        return img_tensor




# -----------------------------
class GrayscaleAugment_aggressive:
    def __init__(self, resize=(224,224), horizontal_flip=True, vertical_flip=True,
                 rotation=15, brightness=0.2, contrast=0.2, normalize=False):
        self.resize = resize
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.brightness = brightness
        self.contrast = contrast
        self.normalize = normalize

    def __call__(self, img_np: np.ndarray):
        """
        img_np: numpy array of shape (H, W), dtype uint16 or float32 in [0,1]
        """

        # Ensure float in [0,1]
        if img_np.dtype == np.uint16:
            img_np = img_np.astype(np.float32) / 65535.0
        elif img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
        else:
            img_np = img_np.astype(np.float32)
            if img_np.max() > 1.0:
                img_np /= img_np.max()

        # Convert to PIL (8-bit, since PIL doesn’t handle float32 well)
        img_pil = Image.fromarray((img_np*255).astype(np.uint8), mode="L")

        # Resize
        img_pil = img_pil.resize(self.resize, resample=Image.BILINEAR)

        # Random flips
        if self.horizontal_flip and random.random() < 0.5:
            img_pil = TF.hflip(img_pil)
        if self.vertical_flip and random.random() < 0.5:
            img_pil = TF.vflip(img_pil)

        # Random rotation
        if self.rotation > 0:
            angle = random.uniform(-self.rotation, self.rotation)
            img_pil = TF.rotate(img_pil, angle, fill=0)

        # Random brightness/contrast jitter
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img_pil = TF.adjust_brightness(img_pil, factor)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img_pil = TF.adjust_contrast(img_pil, factor)

        # Back to tensor (1,H,W) in [0,1]
        img_tensor = TF.to_tensor(img_pil)  # already returns float32 0-1

        if self.normalize:
            # Normalize with ImageNet stats
            mean, std = 0.485, 0.229
            img_tensor = (img_tensor - mean) / std

        return img_tensor



# -----------------------------
# Custom transform for 16-bit grayscale to tensor
# -----------------------------
class ToTensorGrayscale:
    def __call__(self, img):
        # img is numpy HxW in [0,1]
        return torch.from_numpy(img).unsqueeze(0)  # 1,H,W


# -----------------------------
# Model for 1-channel input from scratch
# -----------------------------
class SomiteCounter(nn.Module):
    def __init__(self):
        super().__init__()
        #base = models.resnet18(pretrained=False)
        base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Linear(base.fc.in_features, 2)
        self.model = base


# -----------------------------
# Model for 1-channel input with pretrained weights and transfer learning
# -----------------------------
class SomiteCounter_pt(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Change first conv layer to accept 1-channel input
        old_conv1 = base.conv1
        base.conv1 = nn.Conv2d(1, old_conv1.out_channels,
                               kernel_size=old_conv1.kernel_size,
                               stride=old_conv1.stride,
                               padding=old_conv1.padding,
                               bias=False)

        # Copy pretrained weights by averaging across channels
        base.conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)

        # Replace classifier head (2 outputs: total + defective)
        base.fc = nn.Linear(base.fc.in_features, 2)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in base.parameters():
                param.requires_grad = False
            for param in base.fc.parameters():
                param.requires_grad = True
            for param in base.conv1.parameters():
                param.requires_grad = True

        self.model = base

#network structure:
#conv1 → bn1 → relu → maxpool
#layer1 → layer2 → layer3 → layer4 → avgpool → fc
class SomiteCounter_freeze(nn.Module):
    def __init__(self, unfreeze_layers=("layer3", "layer4"),unfreeze_all=False):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adapt conv1 for grayscale
        old_conv1 = base.conv1
        base.conv1 = nn.Conv2d(
            1, old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False
        )
        base.conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)

        # Replace classifier
        base.fc = nn.Linear(base.fc.in_features, 2)

        if unfreeze_all:
            # Make ALL layers trainable
            for param in base.parameters():
                param.requires_grad = True

        else:
            # Freeze everything first
            for param in base.parameters():
                param.requires_grad = False

            # Always train fc and conv1
            for param in base.fc.parameters():
                param.requires_grad = True
            for param in base.conv1.parameters():
                param.requires_grad = True

            # Unfreeze selected layers (e.g. layer3 + layer4)
            for name, module in base.named_children():
                if name in unfreeze_layers:
                    for param in module.parameters():
                        param.requires_grad = True

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
def get_augmentations(resize=(224,224)):
    return T.Compose([
        T.ToPILImage(mode="F"),  # interpret float32 image
        T.Resize(resize),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor()  # scales back to 0–1 tensor
    ])

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

    #use this model when more images are available
    #model = SomiteCounter().to(device)

    #pretrained model with finetuning (all layers freezed except fc and conv1)    
    #model = SomiteCounter_pt().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=lr)

    #pretrained model with partial finetuning (some layers unfreezed)
    model = SomiteCounter_freeze(unfreeze_layers=("layer3", "layer4")).to(device)
    # Define optimizer with differential learning rates
    params = [
        {"params": model.model.fc.parameters(), "lr": 1e-4},       # head
        {"params": model.model.layer3.parameters(), "lr": 1e-5},   # unfreezed backbone
        {"params": model.model.layer4.parameters(), "lr": 1e-5},
        {"params": model.model.conv1.parameters(), "lr": 1e-5}
    ]

    #to unfreeze all layers
    #model = SomiteCounter_freeze(unfreeze_all=True)
    #params = [
    #    {"params": model.model.fc.parameters(), "lr": 1e-4},   # head
    #    {"params": model.model.parameters(), "lr": 1e-5}       # rest of backbone
    #]

    optimizer = torch.optim.Adam(params)



    criterion = WeightedMSELoss()

    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = 0
    epochs_no_improve = 0

    # Resume if more images available (CAUTION this currently not consistent as I am changing the training dataset, while I should just add more images to the same dataset)
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
        if epoch % visualize_every == 0 and visualize_every > 0:
            sample_img, _, _ = train_dataset[0]
            show_image_comparison(sample_img.numpy().squeeze(), sample_img)

    return model





# -----------------------------
if __name__ == "__main__":
    print("Training starts...")

    # ------------------------
    import argparse
    parser = argparse.ArgumentParser(description="Train Somite Counting Model")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume training if checkpoint exists", default=False)
    parser.add_argument("--visualize_every", type=int, default=-1, help="Visualize sample every N epochs")
    parser.add_argument("--visualize_first", action="store_true", default=False, help="Visualize first epochs")
    args = parser.parse_args()


    #transform = GrayscaleAugment(resize=(224,224), horizontal_flip=True, rotation=10)

    transform = GrayscaleAugment_aggressive(
        resize=(224,224),
        horizontal_flip=True,
        vertical_flip=True,
        rotation=10,
        brightness=0.3,
        contrast=0.3
    )

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

    if args.visualize_first:
        sample_img, _, _ = train_dataset[0]
        show_image_comparison(sample_img.numpy().squeeze(), sample_img)
        
    # ------------------------
    # Train the model
    # ------------------------
    model = train_model(
        train_dataset,
        valid_dataset,
        save_dir="checkpoints",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        resume=args.resume,        # resume if checkpoint exists
        visualize_every=args.visualize_every   # show a sample image every 5 epochs
    )

    print("Training completed.")



