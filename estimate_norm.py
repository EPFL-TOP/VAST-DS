import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_mean_std(img_dir):
    """
    Computes mean and std for a folder of grayscale images.
    Supports uint16 or uint8 images.
    """
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    num_pixels = 0

    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))]

    for fname in tqdm(files, desc="Computing mean/std"):
        path = os.path.join(img_dir, fname)
        img = Image.open(path)
        img_np = np.array(img)

        # Normalize to [0,1] depending on dtype
        if img_np.dtype == np.uint16:
            img_np = img_np.astype(np.float32) / 65535.0
        elif img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
        else:
            img_np = img_np.astype(np.float32)
            if img_np.max() > 1:
                img_np /= img_np.max()

        pixel_sum += img_np.sum()
        pixel_sq_sum += (img_np ** 2).sum()
        num_pixels += img_np.size

    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)

    return mean, std

if __name__ == "__main__":
    train_img_dir = r"D:\vast\training_data\train"
    mean, std = compute_mean_std(train_img_dir)
    print(f"Dataset mean: {mean:.4f}, std: {std:.4f}")
