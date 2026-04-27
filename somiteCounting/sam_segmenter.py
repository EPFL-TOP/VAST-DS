"""Thin wrapper around Meta's `segment_anything` library so the SAM dashboard
in `well_explorer.views` can stay free of model-loading details.

Why a wrapper:
  * Lazy import — `segment_anything` and the checkpoint (~350 MB for vit_b,
    ~2.5 GB for vit_h) are only loaded the first time the SAM dashboard is
    opened. Django startup stays fast.
  * Singleton — the loaded SAM lives on the GPU and we never want to load it
    twice.
  * Graceful fallback — if the package isn't installed or the checkpoint is
    missing, the dashboard shows an explicit "SAM not available" message
    instead of crashing.

Install
-------
    pip install git+https://github.com/facebookresearch/segment-anything.git

then drop a checkpoint somewhere readable. Default expected path is
`checkpoints/sam_vit_b_01ec64.pth` (the smallest official checkpoint).
"""

import os
from typing import List, Optional, Tuple

import numpy as np


# Default expected checkpoint location, overridable from the dashboard.
DEFAULT_SAM_CHECKPOINT = os.environ.get(
    "VAST_SAM_CHECKPOINT",
    "checkpoints/sam_vit_b_01ec64.pth",
)
DEFAULT_SAM_MODEL_TYPE = os.environ.get("VAST_SAM_MODEL_TYPE", "vit_b")


class SAMSegmenter:
    """Loads a SAM (or MedSAM) checkpoint and runs point-prompted inference.

    Use the module-level `get_sam_instance()` to acquire a shared singleton
    rather than constructing this directly.
    """

    def __init__(self, checkpoint_path: str, model_type: str = "vit_b",
                 device: Optional[str] = None):
        # Imports kept local so module import is free.
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self._image_set = False

    def set_image(self, img_np: np.ndarray) -> None:
        """Compute the image embedding once. Call before `segment_at_points()`."""
        # SAM expects HxWx3 uint8. Our YFP frames are 2-D float — broadcast and rescale.
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        if img_np.dtype != np.uint8:
            arr = img_np.astype(np.float32)
            mx = float(arr.max()) if arr.max() > 0 else 1.0
            arr = (arr / mx * 255.0).clip(0, 255).astype(np.uint8)
            img_np = arr
        self.predictor.set_image(img_np)
        self._image_set = True

    def segment_at_points(self, points_xy: List[Tuple[float, float]]) -> List[np.ndarray]:
        """Return one (H, W) bool mask per point prompt.

        Empty input → empty list. Each point is treated as a positive prompt
        (`label=1`) so SAM segments the object the user clicked on.
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() before segment_at_points().")
        if not points_xy:
            return []

        masks: List[np.ndarray] = []
        for x, y in points_xy:
            m, _scores, _logits = self.predictor.predict(
                point_coords=np.array([[x, y]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=False,
            )
            masks.append(np.asarray(m[0], dtype=bool))
        return masks


# --------------------------------------------------------------------------
# Singleton management
# --------------------------------------------------------------------------
_INSTANCE: Optional[SAMSegmenter] = None
_LOAD_ERROR: Optional[str] = None


def get_sam_instance(checkpoint_path: Optional[str] = None,
                     model_type: Optional[str] = None
                     ) -> Tuple[Optional[SAMSegmenter], Optional[str]]:
    """Return (segmenter, error). If loading fails (e.g. missing checkpoint or
    `segment_anything` not installed), `segmenter` is None and `error` is a
    human-readable string the dashboard can display."""
    global _INSTANCE, _LOAD_ERROR
    if _INSTANCE is not None:
        return _INSTANCE, None
    if _LOAD_ERROR is not None:
        return None, _LOAD_ERROR

    ckpt = checkpoint_path or DEFAULT_SAM_CHECKPOINT
    mtype = model_type or DEFAULT_SAM_MODEL_TYPE
    if not os.path.isfile(ckpt):
        _LOAD_ERROR = (f"SAM checkpoint not found at {ckpt}. "
                       f"Download one from "
                       f"https://github.com/facebookresearch/segment-anything"
                       f" or set the VAST_SAM_CHECKPOINT environment variable.")
        return None, _LOAD_ERROR
    try:
        _INSTANCE = SAMSegmenter(ckpt, model_type=mtype)
    except ImportError as e:
        _LOAD_ERROR = (f"`segment_anything` is not installed: {e}. "
                       f"Install it with: "
                       f"pip install git+https://github.com/facebookresearch/segment-anything.git")
    except Exception as e:
        _LOAD_ERROR = f"Could not load SAM model: {e}"
    return _INSTANCE, _LOAD_ERROR


# --------------------------------------------------------------------------
# Per-somite extraction from a list of masks
# --------------------------------------------------------------------------
def extract_per_somite_data(masks: List[np.ndarray],
                            img_shape: Tuple[int, int]) -> List[dict]:
    """Convert one-mask-per-somite into the JSON-shaped list used by
    `DestWellPropertiesPredicted.per_somite_data`.

    Per-somite fields:
      index        — 0-based, ordered by anterior-posterior position
      centroid_x   — pixel x of mask centroid
      centroid_y   — pixel y of mask centroid
      area         — pixel count of the mask
      ap_position  — normalised 0..1 along the AP axis of the union of all
                     masks (0 = leftmost / head after orientation correction,
                     1 = rightmost / tail)
      severity     — defect severity. Default 0 (healthy); UI can edit later.
                     Scale: 0 healthy, 1 mild, 2 moderate, 3 severe.
      comments     — free-text notes per somite. Default empty.
    """
    if not masks:
        return []

    # Union for AP normalisation
    union = np.zeros(img_shape, dtype=bool)
    for m in masks:
        if m.shape != img_shape:
            continue
        union |= m
    if not union.any():
        return []
    ys, xs = np.where(union)
    x_min, x_max = float(xs.min()), float(xs.max())
    span = max(x_max - x_min, 1.0)

    somites: List[dict] = []
    for m in masks:
        if not m.any():
            continue
        ys_m, xs_m = np.where(m)
        cx = float(xs_m.mean())
        cy = float(ys_m.mean())
        area = int(m.sum())
        ap = (cx - x_min) / span
        somites.append({
            "index": -1,                   # filled in after sort
            "centroid_x": cx,
            "centroid_y": cy,
            "area": area,
            "ap_position": float(ap),
            "severity": 0,
            "comments": "",
        })

    somites.sort(key=lambda s: s["ap_position"])
    for new_idx, s in enumerate(somites):
        s["index"] = new_idx
    return somites
