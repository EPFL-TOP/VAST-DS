import os
import json
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from training import SomiteCounter, SomiteCounter_freeze, SomiteCounter_pt  # import your model class

# -----------------------------
# Evaluation helper
# -----------------------------
def load_and_prepare_image(img_path, resize=(224,224)):
    img_raw = np.array(Image.open(img_path)).astype(np.float32)
    img_raw /= img_raw.max()  # scale to 0-1

    img_pil = Image.fromarray((img_raw*65535).astype(np.uint16))
    img_pil = img_pil.resize(resize, resample=Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img_pil).astype(np.float32)/65535.0).unsqueeze(0).unsqueeze(0)
    return img_raw, img_tensor

def show_image_prediction(img_raw, gt_total, gt_def, pred_total, pred_def):
    plt.figure(figsize=(6,6))
    plt.imshow(img_raw, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"GT: total={gt_total}, defective={gt_def}\nPred: total={pred_total:.1f}, defective={pred_def:.1f}")
    plt.show()

# -----------------------------
# Main evaluation function
# -----------------------------
def evaluate_folder(img_dir, label_dir, checkpoint_path, save_csv=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = SomiteCounter().to(device)
    model = SomiteCounter_freeze().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.tif','.tiff','.png'))]
    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        json_path = os.path.join(label_dir, base_name + ".json")

        if not os.path.exists(json_path):
            print(f"Warning: no JSON label for {img_name}, skipping.")
            continue

        # Load image and prepare tensor
        img_raw, img_tensor = load_and_prepare_image(img_path)
        img_tensor = img_tensor.to(device)

        # Load ground truth
        with open(json_path, "r") as f:
            gt = json.load(f)
        gt_total = gt["n_total_somites"]
        gt_def = gt["n_bad_somites"]

        # Prediction
        with torch.no_grad():
            pred = model(img_tensor).cpu().numpy().flatten()
        pred_total, pred_def = pred

        # Display
        show_image_prediction(img_raw, gt_total, gt_def, pred_total, pred_def)

        # Store results
        results.append({
            "image": img_name,
            "gt_total": gt_total,
            "gt_defective": gt_def,
            "pred_total": pred_total,
            "pred_defective": pred_def
        })

    if save_csv:
        pd.DataFrame(results).to_csv(save_csv, index=False)
        print(f"Predictions saved to {save_csv}")

    return results

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    img_dir = r"D:\vast\training_data\valid"
    label_dir = r"D:\vast\training_data\valid"
    checkpoint_path = r"checkpoints/best_model.pth"
    save_csv = "predictions.csv"

    evaluate_folder(img_dir, label_dir, checkpoint_path, save_csv=save_csv)