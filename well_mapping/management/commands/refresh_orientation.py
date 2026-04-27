"""Re-run the orientation classifier on every well and write canonicalised
images into ``<well>/corrected_orientation/``.

This is **Phase A** of the pipeline. Run this whenever the orientation
classifier has been retrained, before re-splitting the training data with
``--use_corrected``.

Wraps the logic in ``somiteCounting.orientfish.OrientationCorrector`` but with
proper CLI flags, no hard-coded paths in the entry point, and an
``--experiment`` filter so you can refresh one experiment at a time.

Examples
--------
Refresh every experiment under the current LOCALPATH::

    python manage.py refresh_orientation

Limit to a single experiment (substring match against the folder name)::

    python manage.py refresh_orientation --experiment VAST_2026-04

Use a specific checkpoint path::

    python manage.py refresh_orientation --checkpoint checkpoints/orientation_best.pth
"""

import os

import numpy as np
from django.core.management.base import BaseCommand, CommandError
from PIL import Image

# Same path probing as well_explorer/views.py
LOCALPATH_HIVE  = r"Y:\raw_data\microscopy\vast\VAST-DS"
LOCALPATH_RAID5 = r"D:\vast\VAST-DS"
LOCALPATH_CH    = "/Users/helsens/Software/github/EPFL-TOP/VAST-DS/data"
DEFAULT_ROOTS   = (LOCALPATH_RAID5, LOCALPATH_HIVE, LOCALPATH_CH)

DEFAULT_CHECKPOINT = "checkpoints/orientation_best.pth"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class Command(BaseCommand):
    help = ("Re-run the orientation classifier on every well and write "
            "canonicalised images into <well>/corrected_orientation/.")

    def add_arguments(self, parser):
        parser.add_argument(
            "--root_path", default=None,
            help=(f"Filesystem root containing the experiment folders. If not "
                  f"set, the first existing path among {DEFAULT_ROOTS} is used."))
        parser.add_argument(
            "--experiment", default=None,
            help="Only refresh experiments whose folder name contains this substring.")
        parser.add_argument(
            "--checkpoint", default=DEFAULT_CHECKPOINT,
            help=f"Path to orientation_best.pth (default: {DEFAULT_CHECKPOINT})")
        parser.add_argument(
            "--dry_run", action="store_true", default=False,
            help="Print what would happen, but don't write any files.")

    # ------------------------------------------------------------------
    def handle(self, *args, **opts):
        # Lazy imports — torch is heavy and we don't want to pay the cost on
        # every `manage.py help` invocation.
        import torch
        from somiteCounting.orientfish import OrientationCorrector
        from somiteCounting._common import preprocess_image

        # Pick the root path
        root = opts["root_path"]
        if root is None:
            for cand in DEFAULT_ROOTS:
                if os.path.isdir(cand):
                    root = cand
                    break
            if root is None:
                raise CommandError(
                    "No --root_path given and none of the default roots exist: "
                    + ", ".join(DEFAULT_ROOTS))
        if not os.path.isdir(root):
            raise CommandError(f"--root_path {root} does not exist")

        ckpt_path = opts["checkpoint"]
        if not os.path.isfile(ckpt_path):
            raise CommandError(f"Orientation checkpoint not found: {ckpt_path}")

        exp_filter = opts["experiment"]
        dry = opts["dry_run"]

        self.stdout.write(self.style.NOTICE(
            f"refresh_orientation: root={root} checkpoint={ckpt_path} "
            f"experiment_filter={exp_filter or '(all)'} dry_run={dry}"))

        # Load the model once
        self.stdout.write("Loading orientation model…")
        oc = OrientationCorrector(ckpt_path)

        n_wells = 0
        n_corrected = 0
        n_no_bf     = 0
        n_no_imgs   = 0

        for exp_name in sorted(os.listdir(root)):
            if "VAST_" not in exp_name:
                continue
            if exp_filter and exp_filter not in exp_name:
                continue
            exp_dir = os.path.join(root, exp_name, "Leica images")
            if not os.path.isdir(exp_dir):
                continue
            self.stdout.write(self.style.NOTICE(f"\n=== {exp_name} ==="))

            for plate in sorted(os.listdir(exp_dir)):
                if "plate" not in plate.lower():
                    continue
                plate_dir = os.path.join(exp_dir, plate)
                if not os.path.isdir(plate_dir):
                    continue
                self.stdout.write(f"  {plate}")

                for well in sorted(os.listdir(plate_dir)):
                    if not well.startswith("Well_"):
                        continue
                    well_dir = os.path.join(plate_dir, well)
                    if not os.path.isdir(well_dir):
                        continue
                    n_wells += 1

                    # Find the BF image (the one without 'YFP' and without 'norm')
                    bf_files = [
                        f for f in os.listdir(well_dir)
                        if f.lower().endswith(IMAGE_EXTS)
                        and "BF" in f and "norm" not in f.lower()
                    ]
                    if not bf_files:
                        n_no_bf += 1
                        continue
                    bf_path = os.path.join(well_dir, bf_files[0])

                    # Run the orientation model
                    img_np = np.array(Image.open(bf_path)).astype(np.float32)
                    tensor = preprocess_image(img_np).unsqueeze(0)
                    if torch.cuda.is_available():
                        tensor = tensor.cuda()
                    flip_choice = oc.correct(tensor)
                    # 0 = no flip, 1 = horizontal, 2 = vertical, 3 = both

                    # Apply the same flip to every image in the well and write
                    # the result into corrected_orientation/.
                    save_dir = os.path.join(well_dir, "corrected_orientation")
                    if not dry and not os.path.isdir(save_dir):
                        os.makedirs(save_dir, exist_ok=True)

                    img_count = 0
                    for f in os.listdir(well_dir):
                        if not f.lower().endswith(IMAGE_EXTS):
                            continue
                        src_path = os.path.join(well_dir, f)
                        if not os.path.isfile(src_path):
                            continue
                        try:
                            img = np.array(Image.open(src_path))
                        except Exception as e:
                            self.stderr.write(f"      [skip {f}] {e}")
                            continue
                        if flip_choice == 1:
                            img = np.flip(img, axis=1)
                        elif flip_choice == 2:
                            img = np.flip(img, axis=0)
                        elif flip_choice == 3:
                            img = np.flip(np.flip(img, axis=1), axis=0)

                        if img.dtype == np.uint8:
                            im_out = Image.fromarray(img, mode="L")
                        elif img.dtype == np.uint16:
                            im_out = Image.fromarray(img, mode="I;16")
                        else:
                            im_out = Image.fromarray(img.astype(np.uint16), mode="I;16")

                        if not dry:
                            im_out.save(os.path.join(save_dir, f))
                        img_count += 1

                    if img_count == 0:
                        n_no_imgs += 1
                    else:
                        n_corrected += 1

                    self.stdout.write(
                        f"    {well}: flip={flip_choice} ({img_count} files)")

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. Wells visited: {n_wells} · canonicalised: {n_corrected} "
            f"· no BF found: {n_no_bf} · no other images: {n_no_imgs}"))
        if dry:
            self.stdout.write(self.style.NOTICE("(dry run — no files written)"))
