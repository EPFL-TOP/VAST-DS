"""Run the profile_v2 multi-head model on every profile_v1 candidate and
write predictions to DestWellPropertiesPredicted under
``model_name='profile_v2'``.

Per-well flow:
  1. Look up the well's profile_v1 prediction (each candidate's algo bbox).
  2. Straighten the YFP image (cached if used elsewhere).
  3. For each candidate, crop a 224x224 patch around the algo bbox
     centre — same patch-extraction logic as ``export_training_set_v2``.
  4. Batch the patches through the model in one ``forward()``.
  5. For each candidate:
       * if argmax of softmax == 'reject' (class 4) → put it in
         ``rejected_candidates`` (audit trail, not displayed by default).
       * else → write the refined bbox + severity to ``somites``.

The resulting profile_v2 row carries:
  * ``n_total_somites`` = number of accepted somites
  * ``n_bad_somites``   = severity > 0 among accepted
  * ``per_somite_data['somites']``              — accepted candidates
  * ``per_somite_data['rejected_candidates']``  — audit list
  * ``per_somite_data['body_length']``          — copied from profile_v1
  * ``per_somite_data['algorithm_params']``     — checkpoint, patch size

Examples
--------
    python manage.py predict_profile_v2
    python manage.py predict_profile_v2 --experiment VAST_2026-05 --plate 2
    python manage.py predict_profile_v2 --checkpoint checkpoints/profile_v2_best.pth
    python manage.py predict_profile_v2 --overwrite          # re-run for all wells
    python manage.py predict_profile_v2 --dry_run --limit 5  # smoke test

Tip — to actually close the missing-tail-somites gap, first re-run
``batch_profile_predict`` with a looser ``--peak_prominence`` (e.g. 0.01)
so more candidates are proposed, then run this command. The reject head
filters out the false positives.
"""

import os
from collections import Counter

import numpy as np
from django.core.management.base import BaseCommand, CommandError


LOCALPATH_HIVE  = r"Y:\raw_data\microscopy\vast\VAST-DS"
LOCALPATH_RAID5 = r"D:\vast\VAST-DS"
LOCALPATH_CH    = "/Users/helsens/Software/github/EPFL-TOP/VAST-DS/data"
DEFAULT_ROOTS   = (LOCALPATH_RAID5, LOCALPATH_HIVE, LOCALPATH_CH)
DEFAULT_CHECKPOINT = "checkpoints/profile_v2_best.pth"
DEFAULT_PATCH_SIZE = 224


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
    files = [f for f in files if 'norm' not in os.path.basename(f).lower()]
    return files[0] if files else None


def _iou(b1, b2):
    """IoU between two [x0,y0,x1,y1] axis-aligned bboxes. Returns 0 if
    either rectangle has zero area or they don't overlap."""
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    a1 = max(1, b1[2] - b1[0]) * max(1, b1[3] - b1[1])
    a2 = max(1, b2[2] - b2[0]) * max(1, b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-9)


def _nms(kept_somites, iou_threshold=0.4):
    """Greedy non-maximum suppression. The bbox-regression head can
    collapse two nearby profile_v1 candidates to nearly the same
    refined position (model agrees with itself for an adjacent pair),
    producing visible duplicates. Sort by confidence desc, keep the
    top, drop later ones whose bbox overlaps the kept ones above the
    threshold. Returns (kept_after_nms, suppressed_indices)."""
    if not kept_somites:
        return [], []
    by_conf = sorted(kept_somites,
                     key=lambda s: -float(s.get('confidence', 0.0)))
    kept_out = []
    suppressed = []
    for s in by_conf:
        bb = s.get('bbox') or []
        if len(bb) != 4:
            continue
        dup = any(_iou(bb, k['bbox']) > iou_threshold for k in kept_out)
        if dup:
            suppressed.append(int(s.get('index', -1)))
        else:
            kept_out.append(s)
    # Re-sort by original index so downstream AP-order code still works.
    kept_out.sort(key=lambda s: int(s.get('index', 0)))
    return kept_out, suppressed


def _extract_patch(straight, center_x, center_y, patch_size):
    """Same patch geometry as export_training_set_v2 — clamped, not padded."""
    sh, sw = straight.shape
    half = patch_size // 2
    px = int(round(center_x - half))
    py = int(round(center_y - half))
    px = max(0, min(sw - patch_size, px))
    py = max(0, min(sh - patch_size, py))
    if sw < patch_size or sh < patch_size:
        return None, None, None
    return straight[py:py + patch_size, px:px + patch_size], px, py


class Command(BaseCommand):
    help = ("Run profile_v2 multi-head model on profile_v1 candidates and "
            "save predictions under model_name='profile_v2'.")

    def add_arguments(self, parser):
        parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
        parser.add_argument("--experiment", default=None,
                            help="Substring filter on experiment name.")
        parser.add_argument("--plate", type=int, default=None,
                            help="Limit to a single plate number.")
        parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
        parser.add_argument("--overwrite", action="store_true", default=False,
                            help="Re-run wells that already have a profile_v2 row.")
        parser.add_argument("--model_version", default='',
                            help="Optional version tag (empty = overwrite same row).")
        parser.add_argument("--limit", type=int, default=None,
                            help="Stop after this many wells (smoke testing).")
        parser.add_argument("--bbox_dilate", type=float, default=1.0,
                            help="Multiplicatively expand the predicted bbox "
                                 "width AND height by this factor before "
                                 "saving. 1.0 = no change (default). 1.5 = "
                                 "50%% larger. Useful to make boxes cover "
                                 "the full chevron, not just the bright "
                                 "spine strip — but won't change what the "
                                 "classifier was trained on, so use for "
                                 "viz only.")
        parser.add_argument("--dry_run", action="store_true", default=False)

    # ------------------------------------------------------------------
    def handle(self, *args, **opts):
        import torch
        import torch.nn.functional as F

        from somiteCounting._common import preprocess_image
        from somiteCounting.tile_crops import straighten_yfp
        from somiteCounting.training_profile_v2 import (
            CLASS_LABELS, NUM_CLASSES, make_profile_v2_model,
        )
        from well_mapping.models import DestWellPropertiesPredicted

        ckpt_path = opts["checkpoint"]
        if not os.path.isfile(ckpt_path):
            raise CommandError(f"Checkpoint not found: {ckpt_path}")

        patch_size = int(opts["patch_size"])
        dry = opts["dry_run"]
        mver = opts["model_version"]
        limit = opts["limit"]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stdout.write(f"Using device: {device}")
        self.stdout.write(f"Loading model from {ckpt_path}…")
        model = make_profile_v2_model(num_severity=NUM_CLASSES, dropout=0.0)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        model = model.to(device).eval()

        # Pull profile_v1 candidates to score.
        qs = DestWellPropertiesPredicted.objects.filter(
            model_name='profile_v1',
            per_somite_data__isnull=False,
        ).select_related('dest_well__well_plate__experiment')
        if opts["experiment"]:
            qs = qs.filter(
                dest_well__well_plate__experiment__name__icontains=opts["experiment"])
        if opts["plate"] is not None:
            qs = qs.filter(dest_well__well_plate__plate_number=opts["plate"])

        existing = set(DestWellPropertiesPredicted.objects.filter(
            model_name='profile_v2', model_version=mver,
        ).values_list('dest_well_id', flat=True))
        self.stdout.write(
            f"Existing profile_v2 rows (version={mver!r}): {len(existing)}")

        stats = Counter()
        for pred_v1 in qs.iterator(chunk_size=200):
            stats['wells_seen'] += 1
            if (limit is not None
                and stats['wells_written'] >= limit):
                break

            dest = pred_v1.dest_well
            if dest.id in existing and not opts["overwrite"]:
                stats['skipped_exists'] += 1
                continue

            psd_v1 = pred_v1.per_somite_data or {}
            somites_v1 = psd_v1.get('somites') or []
            if not somites_v1:
                stats['no_somites_v1'] += 1
                continue

            exp_name = dest.well_plate.experiment.name
            plate_n = dest.well_plate.plate_number
            try:
                col_n = int(dest.position_col)
            except (TypeError, ValueError):
                stats['bad_col'] += 1
                continue

            localpath = _localpath_for(exp_name)
            if localpath is None:
                stats['no_localpath'] += 1
                continue
            yfp = _well_yfp_path(localpath, exp_name, plate_n,
                                 dest.position_row, col_n)
            if yfp is None:
                stats['no_image'] += 1
                continue
            try:
                straight, _ = straighten_yfp(yfp)
            except Exception as e:
                stats['straighten_err'] += 1
                self.stderr.write(self.style.ERROR(
                    f"  [err] {exp_name} {dest.position_row}{col_n:02d}: {e}"))
                continue

            sh, sw = straight.shape

            # Build batch of patches for this well (one forward pass).
            patches = []
            patch_offsets = []   # (px, py)
            valid_v1_indices = []
            for i, s in enumerate(somites_v1):
                bb = s.get('bbox') or []
                if len(bb) != 4:
                    stats['no_bbox_in_v1'] += 1
                    continue
                acx = (bb[0] + bb[2]) / 2.0
                acy = (bb[1] + bb[3]) / 2.0
                patch, px, py = _extract_patch(straight, acx, acy, patch_size)
                if patch is None:
                    stats['patch_oob'] += 1
                    continue
                patches.append(patch)
                patch_offsets.append((px, py))
                valid_v1_indices.append(i)

            if not patches:
                stats['no_valid_patches'] += 1
                continue

            batch = torch.stack([
                preprocess_image(p, resize=(patch_size, patch_size))
                for p in patches
            ]).to(device)
            with torch.no_grad():
                sev_logits, bbox_pred = model(batch)
                sev_probs = F.softmax(sev_logits, dim=1).cpu().numpy()
                bbox_pred = bbox_pred.cpu().numpy()

            kept = []
            rejected = []
            for batch_i, v1_i in enumerate(valid_v1_indices):
                orig = somites_v1[v1_i]
                probs = sev_probs[batch_i]
                pred_class = int(np.argmax(probs))
                px, py = patch_offsets[batch_i]

                if pred_class == NUM_CLASSES - 1:  # reject (class 4)
                    rejected.append({
                        'index':       int(orig.get('index', v1_i)),
                        'centroid_x':  float(orig.get('centroid_x', -1)),
                        'centroid_y':  float(orig.get('centroid_y', -1)),
                        'reject_prob': float(probs[NUM_CLASSES - 1]),
                    })
                    continue

                bb = bbox_pred[batch_i]   # [cx, cy, w, h] in patch fraction
                rcx = bb[0] * patch_size + px
                rcy = bb[1] * patch_size + py
                # Apply the dilation knob (1.0 = identity). Useful to
                # see the bbox cover the full chevron instead of just
                # the bright spine strip — viz-only, the model was
                # trained on tight boxes.
                dilate = float(opts.get('bbox_dilate', 1.0) or 1.0)
                rw  = bb[2] * patch_size * dilate
                rh  = bb[3] * patch_size * dilate
                x0 = max(0, int(round(rcx - rw / 2)))
                x1 = min(sw, int(round(rcx + rw / 2)))
                y0 = max(0, int(round(rcy - rh / 2)))
                y1 = min(sh, int(round(rcy + rh / 2)))

                # Confidence among the 4 non-reject classes only.
                non_reject = probs[:NUM_CLASSES - 1]
                conf = float(non_reject.max() / max(non_reject.sum(), 1e-9))

                # ap_position: fraction along the SPINE (0=head, 1=tail).
                # Use body_length (from profile_v1) as the denominator,
                # NOT image width — fish don't always span the full
                # 2048 px, so dividing by image width distorts the
                # head/tail position and makes cross-fish comparison
                # meaningless. Fall back to image width only if
                # profile_v1's body_length is missing or zero.
                body_len = (psd_v1.get('body_length') if psd_v1 else None)
                if body_len and body_len > 0:
                    ap_pos = float(rcx / body_len)
                elif sw > 0:
                    ap_pos = float(rcx / sw)
                else:
                    ap_pos = float(orig.get('ap_position', -1))

                kept.append({
                    'index':           int(orig.get('index', v1_i)),
                    'centroid_x':      float(rcx),
                    'centroid_y':      float(rcy),
                    'bbox':            [x0, y0, x1, y1],
                    'severity':        pred_class,
                    'severity_probs':  [float(p) for p in probs],
                    'confidence':      conf,
                    'ap_position':     ap_pos,
                    'algo_centroid_x': float(orig.get('centroid_x', -1)),
                })

            # Non-maximum suppression: drop duplicates where the bbox
            # head collapsed two adjacent profile_v1 candidates to nearly
            # identical refined positions. Suppressed candidates are
            # tagged 'nms_duplicate' in the rejected list so the audit
            # trail still records that profile_v1 had a candidate there.
            kept, nms_suppressed_idx = _nms(kept, iou_threshold=0.4)
            for dup_idx in nms_suppressed_idx:
                rejected.append({
                    'index':       dup_idx,
                    'reason':      'nms_duplicate',
                })

            n_kept   = len(kept)
            n_bad    = sum(1 for s in kept if s['severity'] > 0)
            n_reject = len(rejected)
            stats['nms_suppressed'] += len(nms_suppressed_idx)

            if not dry:
                DestWellPropertiesPredicted.objects.update_or_create(
                    dest_well=dest,
                    model_name='profile_v2',
                    model_version=mver,
                    defaults={
                        'n_total_somites': n_kept,
                        'n_bad_somites':   n_bad,
                        'per_somite_data': {
                            'somites':             kept,
                            'rejected_candidates': rejected,
                            'body_length':         psd_v1.get('body_length'),
                            'algorithm_params': {
                                'model_name': 'profile_v2',
                                'checkpoint': os.path.abspath(ckpt_path),
                                'patch_size': patch_size,
                            },
                        },
                    },
                )

            stats['wells_written'] += 1
            stats['total_kept']    += n_kept
            stats['total_reject']  += n_reject

            if stats['wells_written'] % 25 == 0:
                self.stdout.write(
                    f"  …{stats['wells_written']} wells; "
                    f"kept {stats['total_kept']}, "
                    f"rejected {stats['total_reject']}")

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. Wells seen: {stats['wells_seen']}, "
            f"written: {stats['wells_written']}, "
            f"kept: {stats['total_kept']}, "
            f"rejected: {stats['total_reject']}"))
        for k, v in stats.items():
            if k in ('wells_seen', 'wells_written',
                     'total_kept', 'total_reject'):
                continue
            if v:
                self.stdout.write(f"  {k}: {v}")
        if dry:
            self.stdout.write(self.style.NOTICE("(dry run — nothing written)"))
