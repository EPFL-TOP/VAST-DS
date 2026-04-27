"""Re-run all DL predictions on every dest well and write/update
``DestWellPropertiesPredicted`` rows.

This is the **batch** counterpart to the dashboard's "Predict Full Plate"
button. Run it after retraining any of the three models so production
predictions reflect the latest weights. Reads the canonicalised images from
``<well>/corrected_orientation/`` by default — make sure
``manage.py refresh_orientation`` has been run with the same orientation
checkpoint first, otherwise predictions will be on stale flipped images.

Examples
--------
Refresh predictions for every dest well in every experiment::

    python manage.py reinfer

Limit to a single experiment and plate::

    python manage.py reinfer --experiment VAST_2026-04 --plate 2

Override checkpoint paths::

    python manage.py reinfer \\
        --checkpoint_somites checkpoints/somite_counting_best.pth \\
        --checkpoint_validity checkpoints/fish_quality_best.pth \\
        --checkpoint_orientation checkpoints/orientation_best.pth

Dry run::

    python manage.py reinfer --dry_run

Note
----
Predictions are written to ``DestWellPropertiesPredicted`` with
``model_name='resnet_v1'``. Pass ``--model_version <tag>`` to keep multiple
runs side-by-side; the default empty version overwrites the previous row
for the same well.
"""

import glob
import os
import sys

from django.core.management.base import BaseCommand, CommandError

# Same path probing as well_explorer/views.py
LOCALPATH_HIVE  = r"Y:\raw_data\microscopy\vast\VAST-DS"
LOCALPATH_RAID5 = r"D:\vast\VAST-DS"
LOCALPATH_CH    = "/Users/helsens/Software/github/EPFL-TOP/VAST-DS/data"
DEFAULT_ROOTS   = (LOCALPATH_RAID5, LOCALPATH_HIVE, LOCALPATH_CH)

DEFAULT_CHECKPOINTS = {
    "somites":     "checkpoints/somite_counting_best.pth",
    "validity":    "checkpoints/fish_quality_best.pth",
    "orientation": "checkpoints/orientation_best.pth",
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _localpath_for(experiment_name, override=None):
    if override:
        return override
    for cand in DEFAULT_ROOTS:
        if os.path.isdir(os.path.join(cand, experiment_name)):
            return cand
    return None


def _well_image_paths(localpath, experiment_name, plate_number,
                      position_row, position_col, use_corrected):
    pad = position_col if int(position_col) >= 10 else f"0{position_col}"
    well_dir = os.path.join(
        localpath, experiment_name, "Leica images",
        f"Plate {plate_number}", f"Well_{position_row}{pad}",
    )
    if use_corrected:
        well_dir = os.path.join(well_dir, "corrected_orientation")
    if not os.path.isdir(well_dir):
        return None, None

    yfp = bf = None
    for f in sorted(os.listdir(well_dir)):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        if "norm" in f.lower():
            continue
        if "YFP" in f and yfp is None:
            yfp = os.path.join(well_dir, f)
        elif "BF" in f and bf is None:
            bf = os.path.join(well_dir, f)
    return yfp, bf


class Command(BaseCommand):
    help = ("Re-run all DL predictions on every dest well and write/update "
            "DestWellPropertiesPredicted rows.")

    def add_arguments(self, parser):
        parser.add_argument("--experiment", default=None,
                            help="Substring filter on experiment name.")
        parser.add_argument("--plate", type=int, default=None,
                            help="Limit to a single plate number.")
        parser.add_argument("--root_path", default=None,
                            help="Override the LOCALPATH probing.")
        parser.add_argument("--checkpoint_somites",
                            default=DEFAULT_CHECKPOINTS["somites"])
        parser.add_argument("--checkpoint_validity",
                            default=DEFAULT_CHECKPOINTS["validity"])
        parser.add_argument("--checkpoint_orientation",
                            default=DEFAULT_CHECKPOINTS["orientation"])
        parser.add_argument("--no_corrected", action="store_true", default=False,
                            help="Read raw images instead of corrected_orientation/.")
        parser.add_argument("--model_version", default='',
                            help="Optional model_version tag stored alongside the prediction "
                                 "(e.g. checkpoint git hash or date). Default '' overwrites "
                                 "the same row on each run.")
        parser.add_argument("--dry_run", action="store_true", default=False)

    # ------------------------------------------------------------------
    def handle(self, *args, **opts):
        # Lazy imports — torch etc. shouldn't be loaded for `manage.py help`.
        import numpy as np
        import torch
        from PIL import Image

        from somiteCounting._common import preprocess_image
        from somiteCounting.training import (
            FishQualityClassifier, SomiteCounter_freeze,
        )
        from somiteCounting.training_orientation import OrientationClassifier

        from well_mapping.models import (
            DestWellPlate, DestWellPosition,
            DestWellPropertiesPredicted, Experiment,
        )

        for label, path in (
            ("somites",     opts["checkpoint_somites"]),
            ("validity",    opts["checkpoint_validity"]),
            ("orientation", opts["checkpoint_orientation"]),
        ):
            if not os.path.isfile(path):
                raise CommandError(f"Checkpoint missing for {label}: {path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stdout.write(self.style.NOTICE(f"Using device: {device}"))

        # ---- Load models ----
        self.stdout.write("Loading models…")
        m_somites = SomiteCounter_freeze().to(device)
        m_somites.load_state_dict(
            torch.load(opts["checkpoint_somites"], map_location=device,
                       weights_only=False)["model_state_dict"])
        m_somites.eval()

        m_validity = FishQualityClassifier().to(device)
        m_validity.load_state_dict(
            torch.load(opts["checkpoint_validity"], map_location=device,
                       weights_only=False)["model_state_dict"])
        m_validity.eval()

        m_orientation = OrientationClassifier().to(device)
        m_orientation.load_state_dict(
            torch.load(opts["checkpoint_orientation"], map_location=device,
                       weights_only=False)["model_state_dict"])
        m_orientation.eval()

        # ---- Choose dest wells ----
        plate_qs = DestWellPlate.objects.select_related("experiment")
        if opts["experiment"]:
            plate_qs = plate_qs.filter(experiment__name__icontains=opts["experiment"])
        if opts["plate"] is not None:
            plate_qs = plate_qs.filter(plate_number=opts["plate"])

        n_total = n_written = n_skip_no_image = n_errors = 0
        use_corrected = not opts["no_corrected"]
        dry = opts["dry_run"]

        for plate in plate_qs:
            exp = plate.experiment
            localpath = _localpath_for(exp.name, override=opts["root_path"])
            if localpath is None:
                self.stdout.write(self.style.WARNING(
                    f"  [skip plate] {exp.name} P{plate.plate_number}: no LOCALPATH found"))
                continue

            for dest in DestWellPosition.objects.filter(well_plate=plate):
                n_total += 1
                yfp, bf = _well_image_paths(
                    localpath, exp.name, plate.plate_number,
                    dest.position_row, dest.position_col, use_corrected,
                )
                if not yfp and not bf:
                    n_skip_no_image += 1
                    continue

                pred_total = pred_def = None
                pred_valid = None
                pred_orient = None

                try:
                    if yfp:
                        img_np = np.array(Image.open(yfp)).astype(np.float32)
                        t = preprocess_image(img_np).unsqueeze(0).to(device)
                        with torch.no_grad():
                            out = m_somites(t).cpu().numpy().flatten()
                            v_logit = m_validity(t).cpu().numpy().flatten()
                        pred_total = max(0, int(round(float(out[0]))))
                        pred_def   = max(0, int(round(float(out[1]))))
                        pred_valid = bool(1.0 / (1.0 + np.exp(-v_logit[0])) > 0.5)

                    if bf:
                        img_np = np.array(Image.open(bf)).astype(np.float32)
                        t = preprocess_image(img_np).unsqueeze(0).to(device)
                        with torch.no_grad():
                            o_logit = m_orientation(t).cpu().numpy().flatten()
                        pred_orient = bool(1.0 / (1.0 + np.exp(-o_logit[0])) > 0.5)
                except Exception as e:
                    n_errors += 1
                    self.stderr.write(self.style.ERROR(
                        f"  [error] {exp.name} P{plate.plate_number} {dest.position_row}{dest.position_col}: {e}"))
                    continue

                if not dry:
                    # Multi-row schema: one row per (dest_well, model_name,
                    # model_version). For now we always write under
                    # model_name='resnet_v1'; SAM and future models will use
                    # different names so they coexist on the same well.
                    defaults = {}
                    if pred_total is not None:
                        defaults['n_total_somites'] = pred_total
                    if pred_def is not None:
                        defaults['n_bad_somites'] = pred_def
                    if pred_valid is not None:
                        defaults['valid'] = pred_valid
                    if pred_orient is not None:
                        defaults['correct_orientation'] = pred_orient
                    DestWellPropertiesPredicted.objects.update_or_create(
                        dest_well=dest,
                        model_name='resnet_v1',
                        model_version=opts["model_version"],
                        defaults=defaults,
                    )
                n_written += 1

                if n_written % 50 == 0:
                    self.stdout.write(
                        f"  …processed {n_written}/{n_total} wells")

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. Visited {n_total} wells, wrote {n_written}, "
            f"skipped {n_skip_no_image} (no images), {n_errors} errors."))
        if dry:
            self.stdout.write(self.style.NOTICE("(dry run — no DB writes)"))
