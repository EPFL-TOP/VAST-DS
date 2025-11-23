# infer_orientation.py
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
import os

from training_orientation import OrientationClassifier

class OrientationCorrector:
    def __init__(self, checkpoint_path):
        self.model = OrientationClassifier().cuda()
        ckpt = torch.load(checkpoint_path)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def preprocess(self, img_np):
        # convert numpy → tensor(1,224,224)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8), mode="L")
        img_pil = img_pil.resize((224, 224))
        tensor = TF.to_tensor(img_pil).unsqueeze(0)  # 1,1,H,W
        return tensor.cuda()

    def score(self, tensor):
        with torch.no_grad():
            logit = self.model(tensor)
            return torch.sigmoid(logit).item()  # scalar

    def correct(self, img_np):
        """
        img_np: float32 numpy array normalized 0-1
        """
        t = self.preprocess(img_np)
        score0 = self.score(t)

        if score0 >= 0.5:
            return img_np  # already correct

        # Try horizontal flip
        t_h = torch.flip(t, dims=[3])
        score_h = self.score(t_h)

        # Try vertical flip
        t_v = torch.flip(t, dims=[2])
        score_v = self.score(t_v)

        # Try both flips
        t_hv = torch.flip(t_h, dims=[2])
        score_hv = self.score(t_hv)

        # Choose best orientation
        scores = [score0, score_h, score_v, score_hv]
        best = np.argmax(scores)

        if best == 0:
            return img_np
        elif best == 1:
            return np.flip(img_np, axis=1)      # horizontal
        elif best == 2:
            return np.flip(img_np, axis=0)      # vertical
        elif best == 3:
            return np.flip(np.flip(img_np, axis=1), axis=0)




oc = OrientationCorrector(os.path.join("checkpoints","orientation_best.pth"))

image_path = r"D:\vast\VAST_2025-06-10\VAST images"
for folder in os.listdir(image_path):
    folder_path =  os.path.join(image_path, folder)
    if os.path.isdir(folder_path):
        if "plate 1" not in folder or "plate 2" not in folder or "Plate 1" not in folder or "Plate 2" not in folder:
            continue
        for plate in os.listdir(folder_path):
            plate_path = os.path.join(folder_path, plate)
            if not os.path.isdir(plate_path):
                continue
            if "Well_" not in plate:
                continue

            for well in os.listdir(plate_path):
                well_path = os.path.join(plate_path, well)

                for f in os.listdir(well_path):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                        img_path = os.path.join(well_path, f)
                        print(f"Processing {img_path}...")
                        img = np.array(Image.open(img_path)).astype(np.float32)
                        img /= img.max()

                        corrected = oc.correct(img)


                        

