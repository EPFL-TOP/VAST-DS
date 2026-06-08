"""Iteration harness for `profile_analysis.analyze_image`.

The interactive Bokeh dashboard is great for poking one well at a time, but
slow to iterate against when tuning the algorithm. This script runs the
detector on a labelled test set and prints per-image errors + an aggregate
MAE so I can A/B parameter changes quickly without involving the dashboard.

Expects this layout (override with --data-dir)::

    data/profile_test/
    ├── images/
    │   ├── <stem>.tiff      (any extension SciPy/Pillow can read works)
    │   └── ...
    └── ground_truth.json    { "<stem>": {"n_total": N, "n_bad": M, ...}, ... }

Where the stem in ground_truth.json matches the basename (without
extension) of the image. Extra keys in each ground-truth entry are
ignored — feel free to add free-text "notes" for your own bookkeeping.

Usage::

    python -m somiteCounting.test_profile
    python -m somiteCounting.test_profile --save-viz out/
    python -m somiteCounting.test_profile --param peak_prominence=0.02 --param detrend_sigma=40

`--param key=value` lets you override any single keyword of
`analyze_image()` from the command line, so a quick A/B is just a couple
of invocations. Float / int casting is detected automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running as `python somiteCounting/test_profile.py` AND
# `python -m somiteCounting.test_profile`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from somiteCounting._common import preprocess_image  # noqa: E402
from somiteCounting.profile_analysis import DEFAULTS, analyze_image  # noqa: E402


# -------------------------------------------------------------------- I/O

def _load_image_normalised(path: str) -> np.ndarray:
    """Read a TIFF (or anything skimage.io supports) and pass it through
    the same percentile-clip normalisation the dashboard uses."""
    from skimage.io import imread
    raw = imread(path).astype(np.float32)
    if raw.ndim != 2:
        raise ValueError(
            f"{path}: expected a 2-D grayscale image, got shape {raw.shape}")
    # preprocess_image returns a (1, H, W) tensor; bring it back to (H, W).
    return preprocess_image(raw, resize=raw.shape[:2]).numpy()[0]


def _load_ground_truth(json_path: str) -> Dict[str, Dict]:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Ground truth JSON not found: {json_path}")
    with open(json_path) as f:
        gt = json.load(f)
    if not isinstance(gt, dict):
        raise ValueError(f"{json_path}: expected a top-level JSON object")
    return gt


def _resolve_image_path(images_dir: str, stem: str) -> Optional[str]:
    """Find the YFP image for a ground-truth stem.

    Tries, in order:
      <stem>_YFP.tiff/.tif/.png/.jpg/.jpeg
      <stem>.tiff/.tif/.png/.jpg/.jpeg
    so the JSON can use either naming convention. Anything containing
    'BF' or 'norm' is skipped explicitly — the profile detector only
    operates on canonicalised YFP fluorescence.
    """
    candidates: List[str] = []
    for suffix in (f"{stem}_YFP", stem):
        for ext in ("tif", "tiff", "png", "jpg", "jpeg"):
            for p in sorted(glob(os.path.join(images_dir,
                                               f"{suffix}.{ext}"))):
                base = os.path.basename(p).lower()
                if "_bf" in base or "/bf" in base or "norm" in base:
                    continue
                candidates.append(p)
        if candidates:
            return candidates[0]
    return None


# -------------------------------------------------------------------- viz

def _save_visualisation(out_path: str, image: np.ndarray, result: Dict,
                        stem: str, gt_total: Optional[int]) -> None:
    """Write a single PNG showing the straightened image with vertical
    lines at every detected somite (severity-coloured) and a small caption.
    Optional — runs only if matplotlib is importable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available; skipping viz)")
        return

    somites = result["somites"]
    n = len(somites)
    title = f"{stem} — detected {n}"
    if gt_total is not None:
        title += f" / gt {gt_total} (Δ {n - gt_total:+d})"

    straight = result.get("straightened_image", image)
    h, w = straight.shape
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 5), height_ratios=[3, 2], constrained_layout=True)
    ax1.imshow(straight, cmap="gray", aspect="auto")
    sev_colors = ["#1a9850", "#fdae61", "#f46d43", "#a50026"]
    for s in somites:
        ax1.axvline(x=s["centroid_x"], color=sev_colors[s["severity"]],
                    alpha=0.7, linewidth=1.5)
    ax1.set_title(title)
    ax1.set_yticks([])

    prof = result["mean_profile"]
    det = result.get("detrended_mean_profile")
    xs = np.arange(len(prof))
    ax2.plot(xs, prof, label="mean", color="#2196F3")
    if det is not None:
        ax2.plot(xs, det, label="detrended", color="#888",
                 linestyle="--", linewidth=1)
    for s in somites:
        ax2.scatter(s["centroid_x"],
                    prof[int(s["centroid_x"])] if 0 <= s["centroid_x"] < len(prof) else 0,
                    color=sev_colors[s["severity"]], s=30, zorder=5)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlabel("x (AP)")

    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# -------------------------------------------------------------------- main

def _parse_override(arg: str) -> Tuple[str, object]:
    """`'peak_prominence=0.02'` -> ('peak_prominence', 0.02)."""
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"--param expects key=value, got {arg!r}")
    k, v = arg.split("=", 1)
    k = k.strip(); v = v.strip()
    # Best-effort cast: int → float → str.
    try:
        return k, int(v)
    except ValueError:
        pass
    try:
        return k, float(v)
    except ValueError:
        pass
    if v.lower() in {"true", "false"}:
        return k, (v.lower() == "true")
    return k, v


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/profile_test",
                        help="Directory containing images/ and ground_truth.json")
    parser.add_argument("--save-viz", default=None, metavar="OUT_DIR",
                        help="Save per-image PNG visualisations to this folder")
    parser.add_argument("--param", action="append", default=[],
                        type=_parse_override, metavar="KEY=VALUE",
                        help="Override one analyze_image kwarg, e.g. peak_prominence=0.02. "
                             "May be repeated.")
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, "images")
    gt_path    = os.path.join(args.data_dir, "ground_truth.json")
    gt = _load_ground_truth(gt_path)
    if not gt:
        print(f"Ground truth file {gt_path} is empty.")
        return 1

    # Resolve algorithm parameters (DEFAULTS, filtered to actual analyze_image
    # kwargs, then CLI overrides applied on top).
    import inspect
    accepted = set(inspect.signature(analyze_image).parameters.keys())
    params = {k: v for k, v in DEFAULTS.items() if k in accepted}
    overrides: Dict[str, object] = {}
    for k, v in args.param:
        if k not in accepted:
            print(f"WARNING: --param {k} is not an analyze_image() kwarg; ignoring.")
            continue
        overrides[k] = v
        params[k] = v

    print("=" * 78)
    print(f"Profile detector test — {len(gt)} ground-truth entries from {gt_path}")
    if overrides:
        print(f"Param overrides: {overrides}")
    print("=" * 78)
    header = f"{'image':<32}  {'pred':>4}  {'gt':>4}  {'|Δ|':>4}  {'pred_bad':>8}  {'gt_bad':>6}  notes"
    print(header)
    print("-" * len(header))

    diffs_total: List[int] = []
    diffs_bad:   List[int] = []

    if args.save_viz:
        os.makedirs(args.save_viz, exist_ok=True)

    for stem, entry in gt.items():
        gt_total = entry.get("n_total")
        gt_bad   = entry.get("n_bad")
        notes    = entry.get("notes", "")

        # Resolve to the YFP file matching this stem
        path = _resolve_image_path(images_dir, stem)
        if path is None:
            print(f"{stem:<32}  ----  {gt_total or '—':>4}  ----  ----      "
                  f"{gt_bad or '—':>6}  (no image found — looked for "
                  f"{stem}_YFP.* and {stem}.*)")
            continue

        try:
            img = _load_image_normalised(path)
            result = analyze_image(img, **params)
        except Exception as e:
            print(f"  ERROR on {stem}: {e}")
            continue

        somites = result["somites"]
        n_total = len(somites)
        n_bad   = sum(1 for s in somites if s["severity"] > 0)

        if gt_total is not None:
            diffs_total.append(n_total - gt_total)
        if gt_bad is not None:
            diffs_bad.append(n_bad - gt_bad)

        gt_total_s = f"{gt_total}" if gt_total is not None else "—"
        gt_bad_s   = f"{gt_bad}"   if gt_bad   is not None else "—"
        delta_s    = (f"{abs(n_total - gt_total)}" if gt_total is not None else "—")
        print(f"{stem:<32}  {n_total:>4}  {gt_total_s:>4}  {delta_s:>4}  "
              f"{n_bad:>8}  {gt_bad_s:>6}  {notes}")

        if args.save_viz:
            out = os.path.join(args.save_viz, f"{stem}.png")
            _save_visualisation(out, img, result, stem, gt_total)

    print("-" * len(header))
    if diffs_total:
        ds = np.array(diffs_total, dtype=float)
        print(f"total counts  — N={len(ds)}, MAE={np.abs(ds).mean():.2f}, "
              f"RMSE={np.sqrt((ds**2).mean()):.2f}, bias={ds.mean():+.2f}, "
              f"max|Δ|={int(np.abs(ds).max())}")
    if diffs_bad:
        db = np.array(diffs_bad, dtype=float)
        print(f"defective     — N={len(db)}, MAE={np.abs(db).mean():.2f}, "
              f"bias={db.mean():+.2f}")
    if args.save_viz:
        print(f"\nVisualisations saved to {args.save_viz}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
