"""Batch counterpart to the profile dashboard's **Save** button.

Walks every dest well in the DB (or in a chosen experiment/plate), runs
``somiteCounting.profile_analysis.analyze_image`` on its canonicalised YFP
image, and writes the result back as a ``DestWellPropertiesPredicted`` row
with ``model_name='profile_v1'`` — exactly the shape the dashboard would
write if you clicked Save on every well in turn.

Why this exists: the per-somite tile extractor (``extract_somite_tiles``)
needs a ``profile_v1`` row per well to know where each somite's bbox is.
Hand-saving from the dashboard scales to a handful of wells; this command
scales to the whole DB.

Examples
--------
Predict everything with the defaults from ``profile_analysis.DEFAULTS``::

    python manage.py batch_profile_predict

Limit to one experiment / plate::

    python manage.py batch_profile_predict --experiment VAST_2026-05 --plate 2

Override algorithm knobs (same as the dashboard sliders)::

    python manage.py batch_profile_predict \\
        --peak_prominence 0.025 --peak_distance 35 --detrend_sigma 50

Re-predict wells that already have a ``profile_v1`` row::

    python manage.py batch_profile_predict --overwrite

Note
----
Skips wells whose YFP image is missing (the well never ran through the
pipeline, or ``refresh_orientation`` hasn't been run for it yet). These
show up in the stats summary as ``no_image``.
"""

import os
from collections import Counter

from django.core.management.base import BaseCommand

# Same probing as the rest of the project
LOCALPATH_HIVE  = r"Y:\raw_data\microscopy\vast\VAST-DS"
LOCALPATH_RAID5 = r"D:\vast\VAST-DS"
LOCALPATH_CH    = "/Users/helsens/Software/github/EPFL-TOP/VAST-DS/data"
DEFAULT_ROOTS   = (LOCALPATH_RAID5, LOCALPATH_HIVE, LOCALPATH_CH)
PROFILE_MODEL_NAME = "profile_v1"


def _localpath_for(experiment_name, override=None):
    if override:
        return override
    for cand in DEFAULT_ROOTS:
        if os.path.isdir(os.path.join(cand, experiment_name)):
            return cand
    return None


def _well_yfp_path(localpath, experiment_name, plate_number,
                   position_row, position_col, use_corrected):
    """Return the canonicalised YFP image path, or None."""
    import glob as _glob
    try:
        col = int(position_col)
    except (TypeError, ValueError):
        return None
    pad = f"{col:02d}"
    well_dir = os.path.join(
        localpath, experiment_name, "Leica images",
        f"Plate {plate_number}", f"Well_{position_row}{pad}",
    )
    if use_corrected:
        well_dir = os.path.join(well_dir, "corrected_orientation")
    if not os.path.isdir(well_dir):
        return None
    files = _glob.glob(os.path.join(well_dir, "*YFP*.tiff"))
    files = [f for f in files
             if "norm" not in os.path.basename(f).lower()]
    return files[0] if files else None


class Command(BaseCommand):
    help = ("Run profile_analysis on every dest well and save the result as "
            "a profile_v1 prediction. Batch counterpart to the dashboard Save.")

    def add_arguments(self, parser):
        parser.add_argument("--experiment", default=None,
                            help="Substring filter on experiment name.")
        parser.add_argument("--plate", type=int, default=None,
                            help="Limit to a single plate number.")
        parser.add_argument("--root_path", default=None,
                            help="Override the LOCALPATH probing.")
        parser.add_argument("--no_corrected", action="store_true", default=False,
                            help="Read raw images instead of corrected_orientation/.")
        parser.add_argument("--overwrite", action="store_true", default=False,
                            help="Re-predict wells that already have a profile_v1 row "
                                 "(default: skip them).")
        parser.add_argument("--model_version", default="",
                            help="Optional tag stored alongside the prediction. "
                                 "Default '' overwrites the same row on each run.")

        # Algorithm knobs — same five the dashboard sliders expose. Defaults
        # come from profile_analysis.DEFAULTS so a tuned default trickles
        # through automatically.
        parser.add_argument("--n_strips", type=int, default=None)
        parser.add_argument("--peak_prominence", type=float, default=None)
        parser.add_argument("--peak_distance", type=int, default=None)
        parser.add_argument("--smoothing_sigma", type=float, default=None)
        parser.add_argument("--detrend_sigma", type=float, default=None)

        parser.add_argument("--dry_run", action="store_true", default=False)
        parser.add_argument("--limit", type=int, default=None,
                            help="Stop after this many wells (for smoke-testing).")

    # ------------------------------------------------------------------
    def handle(self, *args, **opts):
        # Lazy imports — skimage/scipy are heavy and irrelevant for help text.
        import numpy as np
        from skimage.io import imread

        from somiteCounting._common import preprocess_image
        from somiteCounting.profile_analysis import (
            DEFAULTS as PA_DEFAULTS, analyze_image,
        )
        from well_mapping.models import (
            DestWellPlate, DestWellPosition, DestWellPropertiesPredicted,
        )

        use_corrected = not opts["no_corrected"]
        dry = opts["dry_run"]
        overwrite = opts["overwrite"]
        limit = opts["limit"]
        mver = opts["model_version"]

        # Build the algorithm-param dict by overlaying CLI overrides on top
        # of PA_DEFAULTS (only the five knobs the dashboard exposes).
        knob_keys = ("n_strips", "peak_prominence", "peak_distance",
                     "smoothing_sigma", "detrend_sigma")
        algo_params = {k: PA_DEFAULTS[k] for k in knob_keys}
        for k in knob_keys:
            if opts.get(k) is not None:
                algo_params[k] = opts[k]
        self.stdout.write("Algorithm params: " + ", ".join(
            f"{k}={v}" for k, v in algo_params.items()))

        # Pre-fetch which dest_well_ids already have a profile_v1 row so we
        # can skip them in O(1) rather than one DB hit per well.
        existing_ids = set(DestWellPropertiesPredicted.objects.filter(
            model_name=PROFILE_MODEL_NAME, model_version=mver,
        ).values_list("dest_well_id", flat=True))
        self.stdout.write(
            f"Existing {PROFILE_MODEL_NAME} rows (version={mver!r}): "
            f"{len(existing_ids)}")

        plate_qs = DestWellPlate.objects.select_related("experiment")
        if opts["experiment"]:
            plate_qs = plate_qs.filter(experiment__name__icontains=opts["experiment"])
        if opts["plate"] is not None:
            plate_qs = plate_qs.filter(plate_number=opts["plate"])

        stats = Counter()

        for plate in plate_qs:
            exp = plate.experiment
            localpath = _localpath_for(exp.name, override=opts["root_path"])
            if localpath is None:
                self.stdout.write(self.style.WARNING(
                    f"  [skip plate] {exp.name} P{plate.plate_number}: "
                    f"no LOCALPATH found"))
                stats["no_localpath_plates"] += 1
                continue

            for dest in DestWellPosition.objects.filter(well_plate=plate):
                stats["wells_seen"] += 1
                if limit is not None and stats["wells_written"] >= limit:
                    break

                if dest.id in existing_ids and not overwrite:
                    stats["skipped_exists"] += 1
                    continue

                yfp = _well_yfp_path(localpath, exp.name, plate.plate_number,
                                     dest.position_row, dest.position_col,
                                     use_corrected)
                if yfp is None:
                    stats["no_image"] += 1
                    continue

                try:
                    raw = imread(yfp).astype(np.float32)
                    normed = preprocess_image(raw, resize=raw.shape[:2]).numpy()[0]
                    result = analyze_image(normed, **algo_params)
                except Exception as e:
                    stats["analyze_err"] += 1
                    self.stderr.write(self.style.ERROR(
                        f"  [err] {exp.name} P{plate.plate_number} "
                        f"{dest.position_row}{dest.position_col}: {e}"))
                    continue

                somites = result["somites"]
                n_total = len(somites)
                n_bad = sum(1 for s in somites if s["severity"] > 0)

                if not dry:
                    DestWellPropertiesPredicted.objects.update_or_create(
                        dest_well=dest,
                        model_name=PROFILE_MODEL_NAME,
                        model_version=mver,
                        defaults={
                            "n_total_somites": n_total,
                            "n_bad_somites":   n_bad,
                            "per_somite_data": {
                                "somites":     somites,
                                "body_length": result["body_length"],
                                "algorithm_params": algo_params,
                            },
                        },
                    )
                stats["wells_written"] += 1
                stats["tiles_total"] += n_total

                if stats["wells_written"] % 25 == 0:
                    self.stdout.write(
                        f"  …{stats['wells_written']} wells written "
                        f"({stats['tiles_total']} somites so far)")

            if limit is not None and stats["wells_written"] >= limit:
                self.stdout.write(self.style.NOTICE(
                    f"Reached --limit {limit}; stopping."))
                break

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. Wells seen: {stats['wells_seen']}, "
            f"written: {stats['wells_written']}, "
            f"somites total: {stats['tiles_total']}."))
        for k, v in stats.items():
            if k not in ("wells_seen", "wells_written", "tiles_total") and v:
                self.stdout.write(f"  {k}: {v}")
        if dry:
            self.stdout.write(self.style.NOTICE("(dry run — nothing written)"))
