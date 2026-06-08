"""Crop per-somite bbox tiles from every `profile_v1` prediction in the DB.

The profile detector saves a per-somite `bbox` (in the *straightened* image
coordinates) inside `DestWellPropertiesPredicted.per_somite_data`. This
command:

  1. enumerates every `profile_v1` row (optionally filtered by experiment),
  2. re-straightens the well's canonicalised YFP image with the same spine
     fit the algorithm uses,
  3. crops each somite's bbox (+ user-configurable padding) out of the
     straightened image,
  4. writes one PNG per somite + a single `manifest.json` for the whole
     run that downstream code (annotation tool, classifier training) can
     read directly.

Tiles overlap by design — chevron-shaped somites can't be cleanly cut
without including pieces of their neighbours. Pass `--centre-marker` to
overlay a small cross on each tile so a CNN knows which somite in the
patch to score.

Usage::

    python manage.py extract_somite_tiles
    python manage.py extract_somite_tiles --experiment VAST_2026-05-11
    python manage.py extract_somite_tiles --padding 12 --centre-marker
"""

import json
import os
from collections import Counter
from typing import Dict, List, Optional

from django.core.management.base import BaseCommand, CommandError


# Same probing as the rest of the project
LOCALPATH_HIVE  = r"Y:\raw_data\microscopy\vast\VAST-DS"
LOCALPATH_RAID5 = r"D:\vast\VAST-DS"
LOCALPATH_CH    = "/Users/helsens/Software/github/EPFL-TOP/VAST-DS/data"
DEFAULT_OUTPUT  = "data/somite_tiles"
PROFILE_MODEL_NAME = "profile_v1"


def _localpath_for(experiment_name: str) -> Optional[str]:
    for cand in (LOCALPATH_RAID5, LOCALPATH_HIVE, LOCALPATH_CH):
        if os.path.isdir(os.path.join(cand, experiment_name)):
            return cand
    return None


def _well_yfp_path(localpath: str, experiment_name: str, plate_n: int,
                    row: str, col: int) -> Optional[str]:
    """Return the canonicalised YFP image path for the well, or None."""
    import glob as _glob
    pad = f"{col:02d}"
    well_dir = os.path.join(
        localpath, experiment_name, "Leica images",
        f"Plate {plate_n}", f"Well_{row}{pad}",
        "corrected_orientation",
    )
    if not os.path.isdir(well_dir):
        return None
    candidates = _glob.glob(os.path.join(well_dir, "*YFP*.tiff"))
    candidates = [c for c in candidates
                  if "norm" not in os.path.basename(c).lower()]
    return candidates[0] if candidates else None


class Command(BaseCommand):
    help = ("Extract per-somite tile crops from all profile_v1 predictions, "
            "ready for annotation or classifier training.")

    def add_arguments(self, parser):
        parser.add_argument("--output_dir", default=DEFAULT_OUTPUT,
                            help=f"Output folder (default: {DEFAULT_OUTPUT})")
        parser.add_argument("--experiment", default=None,
                            help="Limit to experiments whose name contains this substring.")
        parser.add_argument("--padding", type=int, default=10,
                            help="Extra pixels to add around each bbox (default: 10).")
        parser.add_argument("--centre-marker", action="store_true",
                            default=False,
                            help="Draw a small cross at the somite centre in each tile, "
                                 "so a downstream CNN can tell which somite to score.")
        parser.add_argument("--overwrite", action="store_true", default=False,
                            help="Overwrite existing tiles (default: skip wells whose "
                                 "tiles already exist).")
        parser.add_argument("--dry_run", action="store_true", default=False)

    # ------------------------------------------------------------------
    def handle(self, *args, **opts):
        # Lazy imports — torch/skimage are heavy and irrelevant for help text.
        from well_mapping.models import DestWellPropertiesPredicted
        from somiteCounting.tile_crops import straighten_yfp, crop_tile

        out_dir = opts["output_dir"]
        exp_filter = opts["experiment"]
        padding = max(0, int(opts["padding"]))
        centre = bool(opts["centre_marker"])
        overwrite = bool(opts["overwrite"])
        dry = bool(opts["dry_run"])

        if not dry:
            os.makedirs(out_dir, exist_ok=True)

        # Pull every profile_v1 prediction with non-empty per-somite data.
        qs = DestWellPropertiesPredicted.objects.filter(
            model_name=PROFILE_MODEL_NAME,
            per_somite_data__isnull=False,
        ).select_related("dest_well__well_plate__experiment")
        if exp_filter:
            qs = qs.filter(dest_well__well_plate__experiment__name__icontains=exp_filter)

        # Upfront diagnostic — most common failure mode is "no predictions
        # saved yet". Print a one-liner of what *is* in the table so the
        # user can tell whether to seed via the dashboard / a batch save.
        n_match = qs.count()
        self.stdout.write(f"Matching profile_v1 rows: {n_match}")
        if n_match == 0:
            from django.db.models import Count
            by_model = (DestWellPropertiesPredicted.objects
                        .values("model_name")
                        .annotate(n=Count("id"))
                        .order_by("-n"))
            if by_model:
                self.stdout.write("DB has predictions for these model_names:")
                for row in by_model:
                    self.stdout.write(f"  {row['model_name']!r}: {row['n']}")
            else:
                self.stdout.write("No prediction rows of any kind in DB.")
            self.stdout.write(self.style.WARNING(
                "Nothing to extract. Run profile detection from the dashboard "
                "(Save button) for the wells you care about, or add a batch "
                "command — see README 'Per-somite tile extraction'."))
            return

        manifest: List[Dict] = []
        stats = Counter()

        for pred in qs.iterator(chunk_size=500):
            dest = pred.dest_well
            plate = dest.well_plate
            exp = plate.experiment.name

            psd = pred.per_somite_data or {}
            somites = psd.get("somites") if isinstance(psd, dict) else None
            if not somites:
                stats["no_somites"] += 1
                continue

            localpath = _localpath_for(exp)
            if localpath is None:
                stats["no_localpath"] += 1
                self.stderr.write(self.style.WARNING(
                    f"  [skip] no LOCALPATH for {exp}"))
                continue

            try:
                col = int(dest.position_col)
            except (TypeError, ValueError):
                stats["bad_col"] += 1
                continue

            yfp_path = _well_yfp_path(localpath, exp, plate.plate_number,
                                       dest.position_row, col)
            if yfp_path is None:
                stats["no_image"] += 1
                continue

            # Output sub-folder per experiment, file name encodes plate+well
            exp_subdir = os.path.join(out_dir, exp)
            stem = f"Plate{plate.plate_number}_{dest.position_row}{col:02d}"
            if not dry:
                os.makedirs(exp_subdir, exist_ok=True)

            # Skip wells whose first tile already exists, unless overwrite
            first_tile = os.path.join(exp_subdir, f"{stem}_somite_000.png")
            if os.path.exists(first_tile) and not overwrite:
                stats["skipped_exists"] += 1
                continue

            try:
                straight, _ = straighten_yfp(yfp_path)
            except Exception as e:
                stats["preprocess_err"] += 1
                self.stderr.write(self.style.ERROR(
                    f"  [err]  {exp} P{plate.plate_number} {dest.position_row}{col:02d}: {e}"))
                continue

            for s in somites:
                img, meta = crop_tile(straight, s, padding=padding,
                                      centre_marker=centre)
                if img is None:
                    continue

                tile_name = f"{stem}_somite_{int(s.get('index', 0)):03d}.png"
                tile_path = os.path.join(exp_subdir, tile_name)
                if not dry:
                    img.save(tile_path)

                bbox = s.get("bbox") or [None, None, None, None]
                manifest.append({
                    "tile_path": os.path.relpath(tile_path, out_dir),
                    "experiment": exp,
                    "plate": plate.plate_number,
                    "well": f"{dest.position_row}{col:02d}",
                    "dest_well_id": dest.id,
                    "somite_index": int(s.get("index", -1)),
                    "ap_position": float(s.get("ap_position", -1)),
                    "bbox_straightened": [int(v) for v in bbox],
                    "bbox_padded": meta["bbox_padded"],
                    "centre_xy_in_tile": meta["centre_xy_in_tile"],
                    "severity_heuristic": int(s.get("severity", 0)),
                    "severity_reason": s.get("severity_reason", ""),
                    "severity_annotated": None,  # see SomiteAnnotation table
                    "confidence": float(s.get("confidence", 0)),
                    "upper_confidence": float(s.get("upper_confidence", 0)),
                    "lower_confidence": float(s.get("lower_confidence", 0)),
                    "intensity": float(s.get("intensity", 0)),
                })
                stats["tiles_written"] += 1

            stats["wells_processed"] += 1

        # Write manifest
        manifest_path = os.path.join(out_dir, "manifest.json")
        if not dry:
            with open(manifest_path, "w") as f:
                json.dump({
                    "schema_version": 1,
                    "options": {
                        "padding": padding,
                        "centre_marker": centre,
                    },
                    "tiles": manifest,
                }, f, indent=2)

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. Wells processed: {stats['wells_processed']}, "
            f"tiles written: {stats['tiles_written']}."))
        for k, v in stats.items():
            if k not in ("wells_processed", "tiles_written") and v:
                self.stdout.write(f"  {k}: {v}")
        if not dry:
            self.stdout.write(f"Manifest: {manifest_path}")
        else:
            self.stdout.write(self.style.NOTICE("(dry run — nothing written)"))
