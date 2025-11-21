import os
import json
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from training import SomiteCounter, SomiteCounter_freeze, SomiteCounter_pt, FishQualityClassifier  # import your model class

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

def show_image_prediction(img_raw, gt_total, gt_total_err, gt_def, gt_def_err, pred_total, pred_def, img_name=""):
    plt.figure(figsize=(6,6))
    plt.imshow(img_raw, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"image name: {img_name}\nGT: total={gt_total}+/-{gt_total_err}, defective={gt_def}+/-{gt_def_err}\nPred: total={pred_total:.1f}, defective={pred_def:.1f}")
    plt.show()



#________________________________________
def evaluate_image(img_file, checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = SomiteCounter().to(device)
    model = SomiteCounter_freeze().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []

    # Load image and prepare tensor
    img_raw, img_tensor = load_and_prepare_image(img_file)
    img_tensor = img_tensor.to(device)


    # Prediction
    with torch.no_grad():
        pred = model(img_tensor).cpu().numpy().flatten()
    pred_total, pred_def = pred


    # Store results
    results={
        "pred_total": pred_total,
        "pred_defective": pred_def
    }

    return results


# -----------------------------
# Main evaluation function
# -----------------------------
def evaluate_folder(img_dir, label_dir, checkpoint_path, save_csv=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = SomiteCounter().to(device)
    model = SomiteCounter_freeze().to(device)
    checkpoint = torch.load(os.path.join(checkpoint_path,"best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()



    model_fish = FishQualityClassifier().to(device)
    checkpoint = torch.load(os.path.join(checkpoint_path,"fish_quality_best.pth"), map_location=device)
    model_fish.load_state_dict(checkpoint["model_state_dict"])
    model_fish.eval()


    #model_fish = torch.load(os.path.join(checkpoint_path,"fish_quality_best.pth"), map_location=device)
    #model_fish.eval()

    results = []

    print(f"Evaluating images in {img_dir} with labels in {label_dir} using model {checkpoint_path} on device {device}")
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
        gt_total_err = gt["n_total_somites_err"]
        gt_def_err = gt["n_bad_somites_err"]
        gt_valid = gt["valid", True]

        # Prediction
        with torch.no_grad():
            pred = model(img_tensor).cpu().numpy().flatten()
        pred_total, pred_def = pred



        with torch.no_grad():
            logit = model(img_tensor)
            prob = torch.sigmoid(logit).item()

        label = "VALID fish" if prob >= 0.5 else "INVALID fish"
        print(f"Image: {img_name} | Fish quality prediction: {label} (prob={prob:.3f})")
        fish_quality = label
        fish_prob = prob

        # Display
        show_image_prediction(img_raw, gt_total, gt_total_err, gt_def, gt_def_err, pred_total, pred_def, img_name=img_name)

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


  # ------------------------
    import argparse
    parser = argparse.ArgumentParser(description="Train Somite Counting Model")
    parser.add_argument("--input_data_path", type=str, default=r"D:\vast\training_data", help="Path to training data")
    parser.add_argument("--model_checkpoint", type=str, help="Path to model checkpoint")
    args = parser.parse_args()


    img_dir=os.path.join(args.input_data_path,"valid")
    label_dir=os.path.join(args.input_data_path,"valid")
    checkpoint_path = args.model_checkpoint
    save_csv = "predictions.csv"

    evaluate_folder(img_dir, label_dir, checkpoint_path, save_csv=save_csv)