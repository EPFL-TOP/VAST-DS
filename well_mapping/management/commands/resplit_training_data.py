"""Re-split annotated wells into train / validation / held-out test sets.

Resets `use_for_training`, `use_for_validation`, `use_for_test` on every
`DestWellProperties` row, then assigns each annotated well randomly to one of
the three subsets (default 70 / 15 / 15) — and rebuilds the on-disk folder
structure (train/, valid/, test/) at `--output_path` to match.

Filenames follow the convention used by the dashboard's "Create Training Set"
button: ``<experiment>_Plate<N>_<row><col>_(YFP|BF).tiff`` plus a JSON sidecar
holding the labels copied from the database.

Examples
--------
Dry run (no DB writes, no file moves) — see what would happen::

    python manage.py resplit_training_data --dry_run

Real run with explicit ratios and a fixed seed::

    python manage.py resplit_training_data \\
        --output_path D:/vast/VAST-DS/training_data \\
        --train 0.70 --valid 0.15 --test 0.15 \\
        --seed 42 \\
        --use_corrected
"""

import json
import os
import random
import shutil
from collections import Counter

from django.core.management.base import BaseCommand, CommandError

from well_mapping.models import (
    DestWellPlate,
    DestWellPosition,
    DestWellProperties,
    Experiment,
)


# Same path-probing list as well_explorer/views.py
LOCALPATH_HIVE  = r"Y:\raw_data\microscopy\vast\VAST-DS"
LOCALPATH_RAID5 = r"D:\vast\VAST-DS"
LOCALPATH_CH    = "/Users/helsens/Software/github/EPFL-TOP/VAST-DS/data"
DEFAULT_OUTPUT  = r"D:\vast\VAST-DS\training_data"


def _localpath_for(experiment_name: str) -> str:
    """Return the LOCALPATH that contains source images for `experiment_name`."""
    for candidate in (LOCALPATH_RAID5, LOCALPATH_HIVE, LOCALPATH_CH):
        if os.path.isdir(os.path.join(candidate, experiment_name)):
            return candidate
    return LOCALPATH_HIVE   # legacy default


def _well_path(localpath: str, experiment_name: str, plate_number: int,
               position_row: str, position_col: str, use_corrected: bool) -> str:
    pad = position_col if int(position_col) >= 10 else f"0{position_col}"
    base = os.path.join(
        localpath, experiment_name, "Leica images",
        f"Plate {plate_number}", f"Well_{position_row}{pad}",
    )
    return os.path.join(base, "corrected_orientation") if use_corrected else base


class Command(BaseCommand):
    help = "Re-split annotated wells into train/valid/test and rebuild the on-disk folders."

    def add_arguments(self, parser):
        parser.add_argument("--output_path", default=DEFAULT_OUTPUT,
                            help=f"Output folder (default: {DEFAULT_OUTPUT})")
        parser.add_argument("--train", type=float, default=0.70, help="Train ratio (default 0.70)")
        parser.add_argument("--valid", type=float, default=0.15, help="Valid ratio (default 0.15)")
        parser.add_argument("--test",  type=float, default=0.15, help="Test ratio (default 0.15)")
        parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
        parser.add_argument("--use_corrected", action="store_true", default=False,
                            help="Pull from <well>/corrected_orientation/ instead of the raw well folder")
        parser.add_argument("--dry_run", action="store_true", default=False,
                            help="Print the plan, don't touch the DB or filesystem")
        parser.add_argument("--keep_invalid", action="store_true", default=False,
                            help="Include `valid=False` annotations (by default they are skipped, "
                                 "matching how training scripts filter their datasets)")

    # ------------------------------------------------------------------
    def handle(self, *args, **opts):
        ratios = (opts["train"], opts["valid"], opts["test"])
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise CommandError(f"Ratios must sum to 1, got {sum(ratios):.4f}")

        out = opts["output_path"]
        seed = opts["seed"]
        dry = opts["dry_run"]
        keep_invalid = opts["keep_invalid"]

        random.seed(seed)
        self.stdout.write(self.style.NOTICE(
            f"Re-split: train={ratios[0]} valid={ratios[1]} test={ratios[2]} "
            f"seed={seed} dry_run={dry} keep_invalid={keep_invalid}"))

        # ---- 1. Collect annotated wells with usable somite labels ----
        candidates = []
        for props in DestWellProperties.objects.select_related(
            "dest_well__well_plate__experiment").iterator():
            if props.n_total_somites is None or props.n_total_somites < 0:
                continue
            if props.n_bad_somites is None or props.n_bad_somites < 0:
                continue
            if not keep_invalid and not props.valid:
                continue
            if props.dest_well is None:
                continue
            candidates.append(props)

        n = len(candidates)
        if n == 0:
            self.stdout.write(self.style.ERROR(
                "No annotated wells found — nothing to do."))
            return

        # ---- 2. Random shuffle, then partition by ratios ----
        random.shuffle(candidates)
        n_train = int(round(n * ratios[0]))
        n_valid = int(round(n * ratios[1]))
        # remainder goes to test so the three counts always sum exactly to n
        n_test  = n - n_train - n_valid
        buckets = (
            ("train", candidates[:n_train]),
            ("valid", candidates[n_train:n_train + n_valid]),
            ("test",  candidates[n_train + n_valid:]),
        )
        self.stdout.write(self.style.NOTICE(
            f"Found {n} eligible annotations → "
            f"{n_train} train / {n_valid} valid / {n_test} test"))

        # ---- 3. Rebuild output folder structure ----
        if not dry:
            if os.path.exists(out):
                self.stdout.write(self.style.WARNING(f"Removing existing {out}"))
                shutil.rmtree(out)
            os.makedirs(out, exist_ok=True)
            for name, _ in buckets:
                os.makedirs(os.path.join(out, name), exist_ok=True)

        # ---- 4. Reset every annotation's flags upfront so anything that was
        #         flagged before but is no longer eligible (or has missing
        #         source images) ends up cleanly with all three flags False. ----
        if not dry:
            DestWellProperties.objects.update(
                use_for_training=False,
                use_for_validation=False,
                use_for_test=False,
            )

        # ---- 5. Set new flags + copy files for each successful candidate. ----
        copied = Counter()
        missing_files = 0
        for bucket_name, props_list in buckets:
            for props in props_list:
                dest = props.dest_well
                exp_name = dest.well_plate.experiment.name
                plate_n  = dest.well_plate.plate_number
                row      = dest.position_row
                col      = dest.position_col

                localpath = _localpath_for(exp_name)
                src_dir   = _well_path(localpath, exp_name, plate_n, row, col,
                                        opts["use_corrected"])

                # Source image lookup — pick the non-norm tiff for each channel
                yfp_src = bf_src = None
                if os.path.isdir(src_dir):
                    for f in os.listdir(src_dir):
                        if not f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                            continue
                        if "norm" in f.lower():
                            continue
                        if "YFP" in f and yfp_src is None:
                            yfp_src = os.path.join(src_dir, f)
                        elif "BF" in f and bf_src is None:
                            bf_src = os.path.join(src_dir, f)

                if not yfp_src or not bf_src:
                    missing_files += 1
                    self.stdout.write(self.style.WARNING(
                        f"  [skip] {exp_name} P{plate_n} {row}{col}: missing YFP/BF in {src_dir}"))
                    continue

                # Update DB flags on this annotation
                props.use_for_training   = (bucket_name == "train")
                props.use_for_validation = (bucket_name == "valid")
                props.use_for_test       = (bucket_name == "test")
                if not dry:
                    props.save()

                # Build target filenames
                stem    = f"{exp_name}_Plate{plate_n}_{row}{col}"
                yfp_dst = os.path.join(out, bucket_name, f"{stem}_YFP.tiff")
                bf_dst  = os.path.join(out, bucket_name, f"{stem}_BF.tiff")
                json_payload = {
                    "n_total_somites":     props.n_total_somites,
                    "n_bad_somites":       props.n_bad_somites,
                    "n_total_somites_err": props.n_total_somites_err,
                    "n_bad_somites_err":   props.n_bad_somites_err,
                    "valid":               props.valid,
                    "correct_orientation": props.correct_orientation,
                    "comments":            props.comments,
                }

                if not dry:
                    shutil.copy(yfp_src, yfp_dst)
                    shutil.copy(bf_src, bf_dst)
                    with open(os.path.join(out, bucket_name, f"{stem}_YFP.json"), "w") as f:
                        json.dump(json_payload, f, indent=2)
                    with open(os.path.join(out, bucket_name, f"{stem}_BF.json"), "w") as f:
                        json.dump(json_payload, f, indent=2)
                copied[bucket_name] += 1

        # ---- Summary ----
        self.stdout.write(self.style.SUCCESS(
            f"Done. Copied per bucket: train={copied['train']}, "
            f"valid={copied['valid']}, test={copied['test']}"))
        if missing_files:
            self.stdout.write(self.style.WARNING(
                f"  ({missing_files} annotation(s) skipped because source images were missing — "
                f"DB flags were NOT changed for those rows)"))
        if dry:
            self.stdout.write(self.style.NOTICE("(dry run — no files written, no DB updates)"))
