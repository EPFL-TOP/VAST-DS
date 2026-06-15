"""Export labelled per-somite tiles + a manifest for severity-classifier training.

Walks ``SomiteAnnotation``, looks up each somite's bbox in the
corresponding ``profile_v1`` row, re-crops via the shared
``somiteCounting.tile_crops.crop_tile`` helper (so the pixels are
byte-identical to what the annotator labelled), and writes one PNG per
somite + a CSV manifest.

By default the export includes only rows that are usable for severity
training:

  * ``box_quality = 'single'``  (no ambiguous boxes)
  * ``severity IS NOT NULL``    (no 'unsure' rows)
  * ``lr_offset = False``       (no imaging-artefact rows — pass
                                 ``--include_lr_offset`` to keep them)

Output layout (class-folder mode, the default)::

    <output_dir>/
    ├── tiles/
    │   ├── 0_healthy/   *.png
    │   ├── 1_mild/      *.png
    │   ├── 2_moderate/  *.png
    │   └── 3_severe/    *.png
    └── manifest.csv

The class-folder layout is ``torchvision.datasets.ImageFolder``-compatible
so the training script can do ``ImageFolder(out_dir/'tiles', …)``
straight into a DataLoader. Pass ``--flat`` to dump every PNG in one
folder if you prefer to split via the manifest.

Examples
--------
    python manage.py export_somite_training_set
    python manage.py export_somite_training_set --annotator clement --output_dir data/training_v1
    python manage.py export_somite_training_set --experiment VAST_2026-05
    python manage.py export_somite_training_set --flat --dry_run
"""

import csv
import os
from collections import Counter

from django.core.management.base import BaseCommand


# Same path probing pattern as the other commands.
LOCALPATH_HIVE  = r"Y:\raw_data\microscopy\vast\VAST-DS"
LOCALPATH_RAID5 = r"D:\vast\VAST-DS"
LOCALPATH_CH    = "/Users/helsens/Software/github/EPFL-TOP/VAST-DS/data"
DEFAULT_ROOTS   = (LOCALPATH_RAID5, LOCALPATH_HIVE, LOCALPATH_CH)
DEFAULT_OUTPUT  = "data/somite_training_set"

CLASS_DIR = {0: '0_healthy', 1: '1_mild', 2: '2_moderate', 3: '3_severe'}


def _localpath_for(experiment_name):
    for cand in DEFAULT_ROOTS:
        if os.path.isdir(os.path.join(cand, experiment_name)):
            return cand
    return None


def _well_yfp_path(localpath, exp_name, plate_n, row, col):
    import glob as _glob
    well_dir = os.path.join(
        localpath, exp_name, "Leica images",
        f"Plate {plate_n}", f"Well_{row}{col:02d}",
        "corrected_orientation",
    )
    if not os.path.isdir(well_dir):
        return None
    files = _glob.glob(os.path.join(well_dir, "*YFP*.tiff"))
    files = [f for f in files
             if 'norm' not in os.path.basename(f).lower()]
    return files[0] if files else None


class Command(BaseCommand):
    help = ("Export labelled per-somite tiles + manifest CSV for the "
            "severity classifier. Uses the same crop_tile helper as the "
            "annotation dashboard, so pixels match what was labelled.")

    def add_arguments(self, parser):
        parser.add_argument("--output_dir", default=DEFAULT_OUTPUT,
                            help=f"Output folder (default: {DEFAULT_OUTPUT}).")
        parser.add_argument("--annotator", default=None,
                            help="Limit to a single annotator.")
        parser.add_argument("--experiment", default=None,
                            help="Substring filter on experiment name.")
        parser.add_argument("--flat", action="store_true", default=False,
                            help="Skip class folders; put every PNG in "
                                 "<output_dir>/tiles/ and rely on the manifest.")
        parser.add_argument("--exclude_lr_offset", action="store_true", default=False,
                            help="Drop rows with lr_offset=True (default: keep — "
                                 "the model needs to learn that L/R-misaligned fish "
                                 "still have real, ratable somites; the asymmetry "
                                 "is imaging noise, not defect signal).")
        parser.add_argument("--include_unsure", action="store_true", default=False,
                            help="Keep severity=NULL rows under box_quality=single "
                                 "(default: drop). Only useful if you want them as a "
                                 "5th 'rejected' class — most trainers should ignore.")
        parser.add_argument("--overwrite", action="store_true", default=False,
                            help="Overwrite existing PNGs (default: skip them).")
        parser.add_argument("--dry_run", action="store_true", default=False)

    # ------------------------------------------------------------------
    def handle(self, *args, **opts):
        from well_mapping.models import (
            SomiteAnnotation, DestWellPropertiesPredicted,
        )
        from somiteCounting.tile_crops import straighten_yfp, crop_tile

        # Base filter for "training-usable" rows.
        qs = SomiteAnnotation.objects.filter(box_quality='single')
        if not opts["include_unsure"]:
            qs = qs.filter(severity__isnull=False)
        if opts["exclude_lr_offset"]:
            qs = qs.filter(lr_offset=False)
        if opts["annotator"]:
            qs = qs.filter(annotator=opts["annotator"])
        if opts["experiment"]:
            qs = qs.filter(
                dest_well__well_plate__experiment__name__icontains=opts["experiment"])
        qs = qs.select_related('dest_well__well_plate__experiment')

        # Group by dest_well so we straighten each YFP once and crop all
        # its labelled somites in one pass — saves a LOT of polyfit work
        # when an annotator has gone through many somites per fish.
        by_well = {}
        for ann in qs.iterator(chunk_size=500):
            by_well.setdefault(ann.dest_well_id, []).append(ann)

        n_total = sum(len(v) for v in by_well.values())
        self.stdout.write(
            f"Selected {n_total} annotation row(s) across "
            f"{len(by_well)} well(s).")
        if n_total == 0:
            return

        out_dir = opts["output_dir"]
        tiles_root = os.path.join(out_dir, 'tiles')
        dry = opts["dry_run"]
        if not dry:
            os.makedirs(tiles_root, exist_ok=True)
            if not opts["flat"]:
                for d in CLASS_DIR.values():
                    os.makedirs(os.path.join(tiles_root, d), exist_ok=True)

        manifest_rows = []
        stats = Counter()

        for dest_id, anns in by_well.items():
            # Sample annotation for well metadata
            dest = anns[0].dest_well
            exp_name = dest.well_plate.experiment.name
            plate_n = dest.well_plate.plate_number
            try:
                col_n = int(dest.position_col)
            except (TypeError, ValueError):
                stats['bad_col'] += 1
                continue

            # Look up the profile_v1 row once per well to find bboxes.
            pred = (DestWellPropertiesPredicted.objects
                    .filter(dest_well=dest, model_name='profile_v1')
                    .first())
            if pred is None or not pred.per_somite_data:
                stats['no_profile_pred'] += len(anns)
                continue
            somite_by_idx = {
                int(s.get('index', -1)): s
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
                somite = somite_by_idx.get(ann.somite_index)
                if somite is None:
                    stats['no_matching_somite'] += 1
                    continue

                # Human-corrected bbox wins over the algorithm's — the
                # annotator dragged it because it was wrong. The bbox is
                # the ONLY field of the somite dict crop_tile uses, so
                # synthesise a new dict to feed it.
                if ann.corrected_bbox:
                    somite = dict(somite, bbox=list(ann.corrected_bbox))
                    stats['corrected_bbox_used'] += 1
                img, _ = crop_tile(straight, somite, centre_marker=False)
                if img is None:
                    stats['bad_bbox'] += 1
                    continue

                stem = (f"{exp_name}_P{plate_n}_"
                        f"{dest.position_row}{col_n:02d}_"
                        f"s{ann.somite_index:03d}_a-{ann.annotator}")
                sev = ann.severity if ann.severity is not None else -1
                if opts["flat"] or sev < 0:
                    rel = f"{stem}.png"
                else:
                    rel = os.path.join(CLASS_DIR[sev], f"{stem}.png")
                tile_path = os.path.join(tiles_root, rel)

                if os.path.exists(tile_path) and not opts["overwrite"]:
                    stats['skipped_exists'] += 1
                else:
                    if not dry:
                        img.save(tile_path)
                    stats['tiles_written'] += 1
                stats[f'sev_{sev}'] += 1

                manifest_rows.append({
                    'tile_path':    os.path.relpath(tile_path, out_dir),
                    'severity':     ann.severity,
                    'lr_offset':    ann.lr_offset,
                    'bbox_corrected': ann.corrected_bbox is not None,
                    'annotator':    ann.annotator,
                    'experiment':   exp_name,
                    'plate':        plate_n,
                    'well':         f"{dest.position_row}{col_n:02d}",
                    'dest_well_id': dest_id,
                    'somite_index': ann.somite_index,
                })

        # Manifest
        if not dry and manifest_rows:
            man_path = os.path.join(out_dir, 'manifest.csv')
            with open(man_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                w.writeheader()
                w.writerows(manifest_rows)
            self.stdout.write(f"Manifest: {man_path}")

        # Summary
        self.stdout.write(self.style.SUCCESS(
            f"\nDone. Tiles written: {stats['tiles_written']}"
            f" (manifest rows: {len(manifest_rows)})"))
        for sev in (0, 1, 2, 3, -1):
            k = f'sev_{sev}'
            if stats[k]:
                label = {0:'healthy', 1:'mild', 2:'moderate',
                         3:'severe', -1:'unsure'}[sev]
                self.stdout.write(f"  {label}: {stats[k]}")
        for k, v in stats.items():
            if not k.startswith('sev_') and k != 'tiles_written' and v:
                self.stdout.write(f"  {k}: {v}")
        if dry:
            self.stdout.write(self.style.NOTICE("(dry run — nothing written)"))
