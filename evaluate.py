import os
import json
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

# --- Import your model definition ---
from training import SomiteCounter   # <-- replace "your_code" with the file where SomiteCounter is defined
from utils import show_image_comparison

def load_model(checkpoint_path, device):
    model = SomiteCounter().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_one(img_path, json_path, checkpoint_path="checkpoints/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    model = load_model(checkpoint_path, device)

    # --- Load image ---
    #transform = T.Compose([
    #    T.Resize((224,224)),
    #    T.ToTensor(),
    #])
    #img = Image.open(img_path).convert("RGB")
    #img_tensor = transform(img).unsqueeze(0).to(device)

    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img)

    # Show comparison
    show_image_comparison(img, img_tensor, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    # Add batch dimension for inference
    img_tensor = img_tensor.unsqueeze(0).to(device)







    # --- Load ground truth ---
    with open(json_path, "r") as f:
        gt = json.load(f)

    gt_total = gt["n_total_somites"]
    gt_def = gt["n_bad_somites"]

    # --- Inference ---
    with torch.no_grad():
        pred = model(img_tensor).cpu().numpy().flatten()

    pred_total, pred_def = pred

    # --- Plot result ---
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"GT: total={gt_total}, defective={gt_def}\n"
        f"Pred: total={pred_total:.1f}, defective={pred_def:.1f}"
    )
    plt.show()


if __name__ == "__main__":
    # Example usage
    img_path = r"D:\vast\training_data\valid\VAST_2025-07-21_Plate1_D2_YFP.tiff"
    json_path = r"D:\vast\training_data\valid\VAST_2025-07-21_Plate1_D2_YFP.json"
    evaluate_one(img_path, json_path)