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

    def correct(self, img_np, img_np_raw=None):
        """
        img_np: float32 numpy array normalized 0-1
        """
        t = self.preprocess(img_np)
        score0 = self.score(t)

        if score0 >= 0.5:
            return img_np_raw  # already correct

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
            return np.flip(img_np_raw, axis=1)      # horizontal
        elif best == 2:
            return np.flip(img_np_raw, axis=0)      # vertical
        elif best == 3:
            return np.flip(np.flip(img_np_raw, axis=1), axis=0)




oc = OrientationCorrector(os.path.join("checkpoints","orientation_best.pth"))

#image_path = r"D:\vast\VAST_2025-06-10\VAST images"
image_path = r"D:\vast"
for exp in os.listdir(image_path):
    exp_path =  os.path.join(image_path, exp, "Leica images")
    print(f"Processing experiment: {exp}")
    if not os.path.isdir(exp_path):
        continue
    if "VAST_" not in exp:
        continue

    for plate in os.listdir(exp_path):
        plate_path = os.path.join(exp_path, plate)
        print(f"  Processing folder: {plate}")
        if "plate 1" not in plate and "plate 2" not in plate and "Plate 1" not in plate and "Plate 2" not in plate:
            continue

        for well in os.listdir(plate_path):
            print(f"   Processing well: {well}")
            well_path = os.path.join(plate_path, well)
            if not os.path.isdir(well_path):
                continue
            if "Well_" not in well:
                continue

            for f in os.listdir(well_path):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    img_path = os.path.join(well_path, f)
                    print(f"     Processing {img_path}...")
                    img = np.array(Image.open(img_path)).astype(np.float32)
                    img_max = img/img.max()

                    corrected = oc.correct(img_max, img)

                    save_path = os.path.join(well_path, "corrected_orientation")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_file = os.path.join(save_path, f)
                    #corrected_img = (corrected * 255).astype(np.uint8)
                    Image.fromarray(corrected).save(save_file)
                    #print(f"Saved corrected image to {save_file}")

            

                        #D:\vast\VAST_2025-06-10\Leica images\Plate 1\Well_E01

