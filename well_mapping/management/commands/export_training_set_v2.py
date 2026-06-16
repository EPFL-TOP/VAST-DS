"""Export wider 224x224 patches + multi-task labels for the profile_v2 model.

Different from ``export_somite_training_set`` in three crucial ways:

1. **Wider patches**. Crops are a fixed 224x224 (configurable) centered on
   the **algorithm's** bbox centre (NOT the human-edited one). This matches
   what the model sees at inference, when only profile_v1's candidates are
   available — the bbox-regression head learns to *move* the box from the
   algorithm's guess to where the somite really is.

2. **Bbox-regression labels**. The annotator's labelled bbox (from
   ``SomiteAnnotation.bbox``) is recorded in patch coordinates as
   ``[cx, cy, w, h]`` normalised to [0, 1]. Smooth-L1 loss against the
   regression head; the model thus learns to refine the bbox from
   image features alone.

3. **'Reject' class for false positives**. Rows with
   ``box_quality='empty'`` (annotator marked the candidate as a detector
   hallucination) become class 4 ('reject'). The model learns to drop these
   at inference, so we can run profile_v1 with a looser ``peak_prominence``
   to catch missed tail somites — the reject head filters out the chaff.

Output layout::

    <output_dir>/
    ├── patches/
    │   ├── exp_X_plateY_A03_s007.png   (224x224, grayscale)
    │   └── ...
    └── manifest_v2.csv
        # tile_path, class, severity, target_cx, target_cy, target_w, target_h,
        # has_bbox_target, lr_offset, bbox_edited, annotator, experiment,
        # plate, well, dest_well_id, somite_index

Default filters:
  * ``box_quality='single' AND severity NOT NULL`` → severity classes 0–3
  * ``box_quality='empty'``                        → class 4 ('reject')
  * Everything else (multiple, mispositioned, unsure single) is dropped.

Examples
--------
    python manage.py export_training_set_v2
    python manage.py export_training_set_v2 --annotator clement --patch_size 256
    python manage.py export_training_set_v2 --experiment VAST_2026-05
"""

import csv
import os
from collections import Counter

import numpy as np
from django.core.management.base import BaseCommand


LOCALPATH_HIVE  = r"Y:\raw_data\microscopy\vast\VAST-DS"
LOCALPATH_RAID5 = r"D:\vast\VAST-DS"
LOCALPATH_CH    = "/Users/helsens/Software/github/EPFL-TOP/VAST-DS/data"
DEFAULT_ROOTS   = (LOCALPATH_RAID5, LOCALPATH_HIVE, LOCALPATH_CH)
DEFAULT_OUTPUT  = "data/profile_v2_training_set"
DEFAULT_PATCH_SIZE = 224

CLASS_NAME = {0: 'healthy', 1: 'mild', 2: 'moderate', 3: 'severe', 4: 'reject'}


def _localpath_for(name):
    for cand in DEFAULT_ROOTS:
        if os.path.isdir(os.path.join(cand, name)):
            return cand
    return None


def _well_yfp_path(localpath, exp_name, plate_n, row, col):
    import glob as _glob
    well_dir = os.path.join(
        localpath, exp_name, "Leica images",
        f"Plate {plate_n}", f"Well_{row}{col:02d}",
        "corrected_orientation")
    if not os.path.isdir(well_dir):
        return None
    files = _glob.glob(os.path.join(well_dir, "*YFP*.tiff"))
    files = [f for f in files
             if 'norm' not in os.path.basename(f).lower()]
    return files[0] if files else None


def _extract_patch(straight, center_x, center_y, patch_size):
    """Crop a fixed-size patch centered (as much as possible) on
    (center_x, center_y). Returns (patch, patch_x0, patch_y0) where
    (patch_x0, patch_y0) is the patch's top-left corner in image
    coordinates. Patches near the image edge are shifted to stay
    in-bounds rather than padded — the somite is still inside the
    patch since the algorithm's bbox center is always inside the
    image."""
    sh, sw = straight.shape
    half = patch_size // 2
    px = int(round(center_x - half))
    py = int(round(center_y - half))
    # Clamp top-left so the patch fits
    px = max(0, min(sw - patch_size, px))
    py = max(0, min(sh - patch_size, py))
    if sw < patch_size or sh < patch_size:
        return None, None, None   # image too small
    return straight[py:py + patch_size, px:px + patch_size], px, py


class Command(BaseCommand):
    help = ("Export 224x224 patches + multi-task labels for the multi-head "
            "profile_v2 model (severity + bbox regression + reject).")

    def add_arguments(self, parser):
        parser.add_argument("--output_dir", default=DEFAULT_OUTPUT,
                            help=f"Output folder (default: {DEFAULT_OUTPUT}).")
        parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE,
                            help=f"Patch size in pixels (default: {DEFAULT_PATCH_SIZE}).")
        parser.add_argument("--annotator", default=None,
                            help="Limit to a single annotator.")
        parser.add_argument("--experiment", default=None,
                            help="Substring filter on experiment name.")
        parser.add_argument("--exclude_lr_offset", action="store_true", default=False,
                            help="Drop rows with lr_offset=True (default: keep).")
        parser.add_argument("--overwrite", action="store_true", default=False,
                            help="Overwrite existing PNGs.")
        parser.add_argument("--dry_run", action="store_true", default=False)

    # ------------------------------------------------------------------
    def handle(self, *args, **opts):
        from PIL import Image

        from well_mapping.models import (
            SomiteAnnotation, DestWellPropertiesPredicted,
        )
        from somiteCounting.tile_crops import straighten_yfp

        patch_size = int(opts["patch_size"])
        out_dir = opts["output_dir"]
        patches_dir = os.path.join(out_dir, 'patches')
        dry = opts["dry_run"]
        if not dry:
            os.makedirs(patches_dir, exist_ok=True)

        # Base filter: usable for multi-task training.
        qs = SomiteAnnotation.objects.filter(
            box_quality__in=('single', 'empty'),
        )
        if opts["exclude_lr_offset"]:
            qs = qs.filter(lr_offset=False)
        if opts["annotator"]:
            qs = qs.filter(annotator=opts["annotator"])
        if opts["experiment"]:
            qs = qs.filter(
                dest_well__well_plate__experiment__name__icontains=opts["experiment"])
        # For 'single' rows we additionally require severity (drop unsure).
        from django.db.models import Q
        qs = qs.filter(
            Q(box_quality='empty') |
            Q(box_quality='single', severity__isnull=False)
        )
        qs = qs.select_related('dest_well__well_plate__experiment')

        # Group by dest_well — re-straightening once per fish dominates I/O.
        by_well = {}
        for ann in qs.iterator(chunk_size=500):
            by_well.setdefault(ann.dest_well_id, []).append(ann)

        n_total = sum(len(v) for v in by_well.values())
        self.stdout.write(
            f"Selected {n_total} annotation row(s) across "
            f"{len(by_well)} well(s).  Patch size = {patch_size} px.")
        if n_total == 0:
            return

        manifest_rows = []
        stats = Counter()

        for dest_id, anns in by_well.items():
            dest = anns[0].dest_well
            exp_name = dest.well_plate.experiment.name
            plate_n = dest.well_plate.plate_number
            try:
                col_n = int(dest.position_col)
            except (TypeError, ValueError):
                stats['bad_col'] += 1
                continue

            # Pull profile_v1's per_somite_data — we need the algorithm's
            # ORIGINAL bbox for the patch center (matches what the model
            # will see at inference, when only profile_v1 candidates are
            # available).
            pred = (DestWellPropertiesPredicted.objects
                    .filter(dest_well=dest, model_name='profile_v1')
                    .first())
            if pred is None or not pred.per_somite_data:
                stats['no_profile_pred'] += len(anns)
                continue
            algo_bbox_by_idx = {
                int(s.get('index', -1)): list(s.get('bbox') or [])
                for s in (pred.per_somite_data.get('somites') or [])
            }

            # Straighten YFP once per well.
            localpath = _localpath_for(exp_name)
            if localpath is None:
                stats['no_localpath'] += len(anns)
                continue
            yfp = _well_yfp_path(localpath, exp_name, plate_n,
                                 dest.position_row, col_n)
            if yfp is None:
                stats['no_image'] += len(anns)
                continue
            try:
                straight, _ = straighten_yfp(yfp)
            except Exception as e:
                stats['straighten_err'] += len(anns)
                self.stderr.write(self.style.ERROR(
                    f"  [err] {exp_name} {dest.position_row}{col_n:02d}: {e}"))
                continue

            for ann in anns:
                algo_bbox = algo_bbox_by_idx.get(ann.somite_index)
                if not algo_bbox or len(algo_bbox) != 4:
                    stats['no_algo_bbox'] += 1
                    continue
                acx = (algo_bbox[0] + algo_bbox[2]) / 2.0
                acy = (algo_bbox[1] + algo_bbox[3]) / 2.0

                patch, px, py = _extract_patch(straight, acx, acy, patch_size)
                if patch is None:
                    stats['patch_oob'] += 1
                    continue

                # Determine class + target_bbox
                if ann.box_quality == 'empty':
                    cls = 4
                    has_bbox = False
                    # Dummy target — masked out in training loss.
                    tcx = tcy = 0.5
                    tw = th_ = 0.1
                else:
                    cls = int(ann.severity)
                    has_bbox = True
                    tbb = list(ann.bbox or [])
                    if len(tbb) != 4:
                        stats['no_target_bbox'] += 1
                        continue
                    # Normalised patch coords (0..1)
                    tcx = ((tbb[0] + tbb[2]) / 2.0 - px) / patch_size
                    tcy = ((tbb[1] + tbb[3]) / 2.0 - py) / patch_size
                    tw  = (tbb[2] - tbb[0]) / patch_size
                    th_ = (tbb[3] - tbb[1]) / patch_size
                    # If the labelled box falls outside the patch, the
                    # patch is too small for this row. Should be rare
                    # (annotator nudges are usually within the algo's
                    # bbox + padding). Skip and count.
                    if not (0.0 <= tcx <= 1.0 and 0.0 <= tcy <= 1.0
                            and 0.0 < tw <= 1.0 and 0.0 < th_ <= 1.0):
                        stats['target_outside_patch'] += 1
                        continue

                # Scale patch to uint8 PNG (per-patch min/max — same as
                # the existing exporter for consistency).
                lo, hi = float(patch.min()), float(patch.max())
                if hi > lo:
                    patch_u8 = ((patch - lo) / (hi - lo) * 255).astype(np.uint8)
                else:
                    patch_u8 = np.zeros_like(patch, dtype=np.uint8)

                stem = (f"{exp_name}_P{plate_n}_"
                        f"{dest.position_row}{col_n:02d}_"
                        f"s{ann.somite_index:03d}_a-{ann.annotator}")
                rel = f"{stem}.png"
                tile_path = os.path.join(patches_dir, rel)
                if os.path.exists(tile_path) and not opts["overwrite"]:
                    stats['skipped_exists'] += 1
                else:
                    if not dry:
                        from PIL import Image
                        Image.fromarray(patch_u8, mode='L').save(tile_path)
                    stats['patches_written'] += 1
                stats[f'class_{CLASS_NAME[cls]}'] += 1

                manifest_rows.append({
                    'tile_path':       os.path.relpath(tile_path, out_dir),
                    'class':           CLASS_NAME[cls],
                    'class_idx':       cls,
                    'has_bbox_target': has_bbox,
                    'target_cx':       round(float(tcx), 6),
                    'target_cy':       round(float(tcy), 6),
                    'target_w':        round(float(tw), 6),
                    'target_h':        round(float(th_), 6),
                    'lr_offset':       ann.lr_offset,
                    'bbox_edited':     ann.bbox_edited,
                    'annotator':       ann.annotator,
                    'experiment':      exp_name,
                    'plate':           plate_n,
                    'well':            f"{dest.position_row}{col_n:02d}",
                    'dest_well_id':    dest_id,
                    'somite_index':    ann.somite_index,
                })

        if not dry and manifest_rows:
            man_path = os.path.join(out_dir, 'manifest_v2.csv')
            with open(man_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                w.writeheader()
                w.writerows(manifest_rows)
            self.stdout.write(f"Manifest: {man_path}")

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. Patches written: {stats['patches_written']} "
            f"(manifest rows: {len(manifest_rows)})"))
        for cls_name in CLASS_NAME.values():
            k = f'class_{cls_name}'
            if stats[k]:
                self.stdout.write(f"  {cls_name}: {stats[k]}")
        for k, v in stats.items():
            if k.startswith('class_') or k == 'patches_written':
                continue
            if v:
                self.stdout.write(f"  {k}: {v}")
        if dry:
            self.stdout.write(self.style.NOTICE("(dry run — nothing written)"))
