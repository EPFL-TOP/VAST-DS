"""Shared per-somite tile cropping — the **single source of truth** for
what the annotator sees and what the future severity classifier is trained
on.

Two callers use these helpers:

  * ``well_mapping/management/commands/extract_somite_tiles.py`` — batch
    PNG dump for offline classifier training.
  * ``well_explorer/views.py::annotate_handler`` — Bokeh dashboard for
    interactive labelling.

Both go through ``straighten_yfp`` + ``crop_tile`` so the pixels the
annotator labels are byte-identical to the pixels the classifier sees at
inference. Don't crop somites anywhere else.
"""

from typing import Dict, Optional, Tuple

import numpy as np


def straighten_yfp(yfp_input):
    """Load (or accept a pre-loaded array) + spine-straighten a YFP image.

    Parameters
    ----------
    yfp_input : str | np.ndarray
        Either a filesystem path to a TIFF/PNG that ``skimage.io.imread`` can
        read, or an already-loaded 2-D float32 array in the same normalised
        space as ``somiteCounting._common.preprocess_image`` produces.

    Returns
    -------
    straight : np.ndarray (H, W) float32
        Spine-straightened image, ready for bbox cropping.
    y_spine : np.ndarray (W,)
        The per-column spine-y values (handy for the dashboard's centerline
        overlay; the extractor ignores it).
    """
    # Lazy imports — keep skimage/scipy out of `manage.py help`.
    from somiteCounting._common import preprocess_image
    from somiteCounting.profile_analysis import (
        find_spine_centerline, straighten_image,
    )

    if isinstance(yfp_input, np.ndarray):
        normed = yfp_input.astype(np.float32, copy=False)
    else:
        from skimage.io import imread
        raw = imread(yfp_input).astype(np.float32)
        normed = preprocess_image(raw, resize=raw.shape[:2]).numpy()[0]

    y_spine, _ = find_spine_centerline(normed)
    straight, _ = straighten_image(normed, y_spine)
    return straight, y_spine


def crop_tile(straight: np.ndarray, somite: Dict, *,
              padding: int = 10,
              centre_marker: bool = False):
    """Crop a single somite tile from an already-straightened image.

    Parameters
    ----------
    straight : np.ndarray (H, W)
        Output of ``straighten_yfp``.
    somite : dict
        One entry from ``per_somite_data['somites']`` — must contain
        ``'bbox'`` as ``[x0, y0, x1, y1]`` in straightened-image
        coordinates, and (if ``centre_marker``) ``centroid_x`` / ``centroid_y``.
    padding : int
        Extra pixels on every side. Clamped to image bounds.
    centre_marker : bool
        If True, draws a small red cross at the somite's centroid (in the
        cropped frame). Useful so a CNN can tell which somite in the patch
        is the labelled one — neighbours' chevron arms always intrude.

    Returns
    -------
    img : PIL.Image
        Mode ``'L'`` (grayscale) by default, ``'RGB'`` if ``centre_marker``.
    meta : dict
        ``{'bbox_padded': [x0p, y0p, x1p, y1p], 'centre_xy_in_tile': [cx, cy] | None}``.
        Useful for the manifest / dashboard overlays.
    """
    from PIL import Image, ImageDraw

    bbox = somite.get("bbox")
    if not bbox or len(bbox) != 4:
        return None, {"bbox_padded": None, "centre_xy_in_tile": None}

    sh, sw = straight.shape
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0p = max(0, x0 - padding)
    y0p = max(0, y0 - padding)
    x1p = min(sw, x1 + padding)
    y1p = min(sh, y1 + padding)
    if x1p <= x0p or y1p <= y0p:
        return None, {"bbox_padded": None, "centre_xy_in_tile": None}

    patch = straight[y0p:y1p, x0p:x1p]
    if patch.size == 0:
        return None, {"bbox_padded": None, "centre_xy_in_tile": None}

    # Per-tile contrast stretch → uint8 PNG. The classifier will see exactly
    # this, so the annotator must too. Don't stretch globally — neighbouring
    # tiles in a defective region would all look identical.
    lo, hi = float(patch.min()), float(patch.max())
    if hi > lo:
        patch_u8 = ((patch - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        patch_u8 = np.zeros_like(patch, dtype=np.uint8)

    img = Image.fromarray(patch_u8, mode="L")
    cxy: Optional[Tuple[int, int]] = None
    if centre_marker:
        cx = int(somite.get("centroid_x", (x0 + x1) / 2)) - x0p
        cy = int(somite.get("centroid_y", (y0 + y1) / 2)) - y0p
        cxy = (cx, cy)
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        arm = 4
        draw.line([(cx - arm, cy), (cx + arm, cy)],
                  fill=(255, 64, 64), width=1)
        draw.line([(cx, cy - arm), (cx, cy + arm)],
                  fill=(255, 64, 64), width=1)

    return img, {
        "bbox_padded": [x0p, y0p, x1p, y1p],
        "centre_xy_in_tile": list(cxy) if cxy is not None else None,
    }
