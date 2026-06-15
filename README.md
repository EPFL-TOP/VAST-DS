# VAST-DS

High-throughput zebrafish drug-screening tool for the UPOATES lab. Combines:
- a Django web app (well-plate mapping, image browsing, statistics)
- Bokeh dashboards for interactive review and annotation
- PyTorch deep-learning models that score VAST-microscope images for fish
  validity, somite counts, and orientation

---

## Project layout

```text
VAST-DS/
├── VAST_DS/                Django project (settings, root URLs)
├── well_mapping/           Django app — experiments, plates, drugs, wells
├── well_explorer/          Django app — image dashboards, stats, drug plots
├── somiteCounting/         PyTorch training & inference for the 3 models
│   ├── _common.py          Shared preprocessing, dataset base, train loop
│   ├── training.py         Somite-count regressor (2 outputs)
│   ├── training_valid.py   Fish-validity classifier
│   ├── training_orientation.py  Orientation classifier
│   ├── evaluate.py         One-off image / folder evaluation utilities
│   └── orientfish.py       Inference helpers used by the dashboard
├── templates/              Django + Bokeh embed templates
├── static/                 CSS, logos, JS
└── manage.py
```

---

## Running the dev server

```bash
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

The landing page at `/` lists every available view (well-mapping dashboard,
well-explorer dashboard, experiment / drug listings, statistics, drug plot).
Bokeh apps are auto-loaded via `bokeh_django` — see `VAST_DS/urls.py` for the
registration list.

---

## Web pages (overview)

| URL                                 | What it does                                                  |
|-------------------------------------|---------------------------------------------------------------|
| `/`                                 | Landing page with cards linking to every tool                 |
| `/well_mapping/bokeh_dashboard`     | Define / edit drug-to-source-to-destination well mappings     |
| `/well_explorer/bokeh_dashboard`    | Browse images, annotate, run predictions                      |
| `/well_explorer/experiment_listing` | Per-experiment summary table (efficiency, fish counts)        |
| `/well_explorer/drugs_listing`      | Drug-by-drug image gallery with somite stats                  |
| `/well_explorer/stats_listing`      | Plate filling / VAST efficiency / annotation stats / pred-vs-ann comparison |
| `/well_explorer/drug_plot`          | Interactive histograms + predicted-vs-annotated scatters per drug |

---

## Deep-learning models

Three independent models, all ResNet18 with 1-channel input, all sharing the
same preprocessing (1st/99th percentile clip + resize to 224×224, defined once
in `somiteCounting/_common.preprocess_image`).

| Model                  | Input        | Output                                  | Loss     |
|------------------------|--------------|------------------------------------------|----------|
| `SomiteCounter_freeze` | YFP image    | `(n_total_somites, n_bad_somites)` float | Weighted MSE |
| `FishQualityClassifier`| YFP image    | `valid` logit                            | BCE      |
| `OrientationClassifier`| BF image     | `correct_orientation` logit              | BCE      |

Inference predictions are cast to non-negative integers (`clamp_somite_count`)
before being saved to `DestWellPropertiesPredicted`.

### Training data folder layout

Each training script expects:

```text
<input_data_path>/
├── train/
│   ├── img001_YFP.png
│   ├── img001_YFP.json
│   ├── img002_YFP.png
│   ├── img002_YFP.json
│   └── ...
├── valid/
│   └── ... (same shape — used for early stopping)
└── test/
    └── ... (held-out, never seen during training; used only for the final report)
```

Each `.json` sidecar contains the labels for its image. Required keys depend
on the task:

- **Somite counter** (filename contains `YFP`):
  ```json
  {
    "valid": true,
    "n_total_somites": 28,
    "n_bad_somites": 2,
    "n_total_somites_err": 1,
    "n_bad_somites_err": 1
  }
  ```
  Only `valid: true` rows are used. The `*_err` fields down-weight uncertain
  labels in `WeightedMSELoss`.

- **Fish-validity classifier** (filename contains `YFP`):
  ```json
  { "valid": true }
  ```

- **Orientation classifier** (filename does **not** contain `YFP` — it trains
  on brightfield images):
  ```json
  { "valid": true, "correct_orientation": true }
  ```

To build / rebuild the train / valid / test split there's a Django
management command that does the work in one shot — DB flags and on-disk
folders are kept in sync:

```bash
# Dry run first to see what it will do
python manage.py resplit_training_data --dry_run

# Real run
python manage.py resplit_training_data \
    --output_path D:\vast\VAST-DS\training_data \
    --train 0.70 --valid 0.15 --test 0.15 \
    --seed 42 \
    --use_corrected
```

The command resets every annotation's `use_for_training` /
`use_for_validation` / `use_for_test` flags, randomly partitions the
eligible annotations into the three buckets (default 70 / 15 / 15), sets
the corresponding flag on `DestWellProperties`, and copies source images
+ JSON sidecars into the chosen subfolder of `--output_path`. By default
`valid=False` annotations are skipped (pass `--keep_invalid` to include
them). The wells that land in the test bucket show up in the
**Test (honest)** column on the Statistics page.

### Retraining

> **Order matters**: orientation first, then refresh the canonicalised
> images, then re-split, then the count + validity models, then refresh
> production predictions. Each step depends on the canonical output of the
> previous one.

> **Strict rule on invalid fish**: annotations with `valid=False` are
> **never** used in training of any task. `resplit_training_data` and the
> dashboard's "Create Training Set" button both enforce this.

```bash
# 1. Orientation classifier (BF images, no flips during augmentation)
python -m somiteCounting.training_orientation \
    --input_data_path D:\vast\VAST-DS\training_data \
    --model_save_path checkpoints \
    --epochs 40 --batch_size 8

# 2. Refresh corrected_orientation/ for every well, using the new model.
#    Walks each experiment, picks the best of {no flip, hflip, vflip, hflip+vflip}
#    on the BF image, and writes flipped copies to <well>/corrected_orientation/.
python manage.py refresh_orientation \
    --checkpoint checkpoints/orientation_best.pth
# add --experiment VAST_2026-04 to limit to one experiment

# 3. Re-split annotations into train / valid / test and rebuild the on-disk
#    training folders from the freshly canonicalised images. Also sets the
#    use_for_training / use_for_validation / use_for_test flags on
#    DestWellProperties so the Statistics page matches the on-disk split.
#    Includes annotations that have any usable label (somite OR orientation),
#    so orientation-only labels still get fair coverage.
python manage.py resplit_training_data \
    --output_path D:\vast\VAST-DS\training_data \
    --train 0.70 --valid 0.15 --test 0.15 \
    --seed 42 \
    --use_corrected

# 4. Somite counter (regression, 2 outputs — YFP images)
python -m somiteCounting.training \
    --input_data_path D:\vast\VAST-DS\training_data \
    --model_save_path checkpoints \
    --epochs 150 --batch_size 8 --patience 7

# 5. Fish-validity classifier (YFP images)
python -m somiteCounting.training_valid \
    --input_data_path D:\vast\VAST-DS\training_data \
    --model_save_path checkpoints \
    --epochs 50 --batch_size 8

# 6. Refresh production predictions for every dest well in the DB with the
#    new checkpoints (batch counterpart to "Predict Full Plate").
python manage.py reinfer
# limits: --experiment VAST_2026-04 --plate 2

# 7. Restart the Django dev server so the dashboard's in-memory models
#    pick up the new checkpoints.
```

Each training script also runs as `python somiteCounting/<file>.py …` if
you prefer the old style — both work.

After training, three files appear in `checkpoints/`:

```text
checkpoints/
├── somite_counting_best.pth         # SomiteCounter_freeze weights
├── fish_quality_best.pth            # FishQualityClassifier weights
├── orientation_best.pth             # OrientationClassifier weights
├── test_metrics.json                # held-out report for the somite counter
├── fish_quality_test_metrics.json   # held-out report for validity
└── orientation_test_metrics.json    # held-out report for orientation
```

`test_metrics.json` is what you should quote when comparing models. For the
somite counter it contains MAE / RMSE / signed-bias / R² for both raw and
integer-clamped predictions; for the classifiers it contains accuracy /
precision / recall / F1 / TP-FP-FN-TN.

The dashboard loads these checkpoints at startup; restart the Django dev
server after a retrain so the new weights are picked up. Checkpoint paths are
hard-coded in `well_explorer/views.py` (lines ~50–75) — if you change
`--model_save_path`, update them too.

### Why you might want to retrain right now

Until recently each script had its own image normalisation: the somite
counter and validity classifier used `img /= img.max()` (one hot pixel ruins
the contrast), the orientation classifier used percentile-clipping (robust),
and the dashboard inference did yet a third variant. After the
`_common.preprocess_image` consolidation, all three models share the
percentile-clipping path during both training and inference. Existing
checkpoints will still **load** (the architecture didn't change) but their
accuracy may drift because they were trained on a different colour space.
Retraining once on the unified pipeline establishes a clean baseline going
forward.

### Held-out evaluation only (skip training)

The held-out evaluation step runs automatically at the end of each training
run. To re-run it on an existing checkpoint without retraining, load the
checkpoint and call the evaluator directly — see
`somiteCounting/evaluate.py` for the somite counter, or replicate the pattern
from the bottom of each `training_*.py` for the classifiers.

---

## Predictions schema

`DestWellPropertiesPredicted` is keyed by
`(dest_well, model_name, model_version)` — multiple models can store
predictions for the same well side-by-side. Read with
`latest_prediction(dest_well, model_name='resnet_v1')` from
`well_mapping.models`; write with
`DestWellPropertiesPredicted.objects.update_or_create(...)` passing all
three keys.

| Field             | Purpose                                                      |
|-------------------|--------------------------------------------------------------|
| `model_name`      | e.g. `'resnet_v1'`, `'sam_v1'`, `'multitask_v1'`             |
| `model_version`   | optional tag (git hash / date). Empty = overwrite-on-rerun   |
| `predicted_at`    | auto-updated; used to pick the latest row per (well, model)  |
| `per_somite_data` | JSONField — list of `{index, centroid_x, centroid_y, area, ap_position, severity, comments}` for SAM-style segmentation outputs |

Severity scale: 0 = healthy, 1 = mild, 2 = moderate, 3 = severe.
AP position is normalised so 0 = head, 1 = tail.

The migration that introduced this schema is
`well_mapping/migrations/0036_predictions_multirow.py`. Existing rows kept
their data and got `model_name='resnet_v1'` from the field default;
`model_version` stayed empty (same as new rows from the dashboard's
"Predict Full Plate" button).

## SAM segmentation dashboard

Per-somite segmentation page at `/well_explorer/sam_dashboard`. Click each
somite to seed a point prompt, press **Segment**, then **Save** to write a
prediction with `model_name='sam_v1'` and the per-somite list in
`per_somite_data`.

Setup (lazy-loaded on first click):

```bash
# 1. Install Meta's SAM library
pip install git+https://github.com/facebookresearch/segment-anything.git

# 2. Drop a checkpoint at the default path (or set VAST_SAM_CHECKPOINT)
# Smallest official weights are sam_vit_b (~350 MB).
# https://github.com/facebookresearch/segment-anything
cp ~/Downloads/sam_vit_b_01ec64.pth checkpoints/
```

Set `VAST_SAM_CHECKPOINT` to override the path or `VAST_SAM_MODEL_TYPE` to
use a different ViT size (`vit_b` / `vit_l` / `vit_h`). MedSAM weights work
through the same code path; drop them at the same default location.

## Per-somite tile extraction (for the defect classifier)

The current per-somite severity heuristic in `profile_v1` compares each
somite's confidence to its neighbours (±3). That works for the easy cases —
healthy fish and isolated defects — but breaks on uniformly-defective fish
(test MAE ≈ 9 on the bad column) where there's no clean baseline to compare
to. Replacing the heuristic with a small CNN trained on labelled per-somite
crops is the planned fix.

The pipeline is two commands.

**Step 1 — `batch_profile_predict`**: the dashboard's Save button writes a
`profile_v1` row for one well at a time, which doesn't scale. This batch
command walks every dest well in the DB (optionally filtered by
experiment/plate), runs `profile_analysis.analyze_image` on its YFP image,
and persists the result with the same shape Save would write.

```bash
# Predict everything with the tuned DEFAULTS from profile_analysis
python manage.py batch_profile_predict

# Limit to one experiment and one plate, override the prominence knob
python manage.py batch_profile_predict \
    --experiment VAST_2026-05 --plate 2 \
    --peak_prominence 0.025 --peak_distance 35

# Re-predict wells that already have a profile_v1 row (after a tuning change)
python manage.py batch_profile_predict --overwrite

# Quick smoke test — first 10 wells, don't write
python manage.py batch_profile_predict --limit 10 --dry_run
```

By default it **skips** wells that already have a `profile_v1` row, so
incremental runs are cheap; pass `--overwrite` after a meaningful parameter
change. Use `--model_version <tag>` to keep multiple parameter sweeps
side-by-side instead of overwriting.

**Step 2 — `extract_somite_tiles`**: walks every `profile_v1` prediction in
the DB, re-straightens the well's canonicalised YFP image with the same
spine fit `profile_analysis` uses, then crops each somite's `bbox` (with
configurable padding) into a PNG.

```bash
# All experiments, default 10 px padding, no marker
python manage.py extract_somite_tiles

# One experiment, bigger padding, with a centre-cross overlay so the
# downstream classifier knows which somite in the patch to score
python manage.py extract_somite_tiles \
    --experiment VAST_2026-05-11 \
    --padding 12 --centre-marker
```

Output layout:

```text
data/somite_tiles/
├── <experiment>/
│   ├── Plate1_A03_somite_000.png
│   ├── Plate1_A03_somite_001.png
│   └── ...
└── manifest.json
```

`manifest.json` carries the full provenance per tile (`experiment`,
`dest_well_id`, `somite_index`, `ap_position`, `bbox_straightened`,
`bbox_padded`, `severity_heuristic`, `confidence` triple, …) plus a
`severity_annotated: null` slot for the annotation tool to fill in.

Tiles overlap by design — chevron-shaped somites can't be cleanly cut
without including pieces of their neighbours. The `--centre-marker` flag
draws a small red cross at the central somite's centroid so a CNN can tell
which one in the patch to score. Whether the overlap helps (extra spatial
context) or hurts (label noise from neighbours) is itself a question the
classifier training will answer.

**Step 3 — Annotate (Bokeh dashboard)**: `/well_explorer/annotate_somites`
walks every saved `profile_v1` somite in an experiment and presents them
one at a time, **sorted defective-first** (by `n_bad_somites` desc) so the
interesting cases are front-loaded. The tile shown is cropped on-the-fly
via the shared `somiteCounting/tile_crops.py` helper using
`DEFAULT_PADDING=30` — byte-identical to what `extract_somite_tiles`
writes and what the classifier will see at inference, so annotator and
classifier never disagree about what they're looking at.

Per-somite actions, in priority order:

- **Severity** `0` healthy / `1` mild / `2` moderate / `3` severe — the
  primary label. Use this when the box cleanly contains one somite.
- **Unsure** — saves with `severity=NULL, box_quality='single'`. Keeps the
  row so we don't re-show it; training skips NULL severities.
- **Bad bbox** group: `multi` / `empty` / `mispositioned`. Stored in the
  `box_quality` column with `severity=NULL`. Training filters to
  `box_quality='single'`; the other rows feed a future detector audit.
- **Mark rest of fish healthy** — bulk-write `severity=0` for every
  un-annotated somite in the current fish, then jump to the next fish. A
  one-click escape hatch for the 27 healthy somites of a clean fish.
- **Next fish →** — skip remaining un-annotated somites in this fish
  without writing anything.

The header shows two counters: `N / M somites left in this queue` (global)
and `Fish progress: K/T annotated` (per-fish, queried live so it reflects
prior sessions too). If you navigate back onto a somite you've already
labelled, the prior label is displayed and a re-click overwrites it.

Labels go to `SomiteAnnotation` keyed by `(dest_well, somite_index,
annotator)`. Re-loading the same experiment under the same annotator name
skips tiles already done, so the workflow is resumable.

**What the classifier actually trains on.** The tight red box is the
algorithm's bbox; the dashed orange box is `bbox + DEFAULT_PADDING=30 px`
on each side — the *tile crop area*. Training uses the dashed region, not
the tight bbox. The padding gives the CNN enough surrounding context
(neighbour-somite chevron arms, muscle boundaries) to distinguish a real
missing chunk from a normal valley between two somites. Annotator and
classifier import the same `DEFAULT_PADDING` constant from
`somiteCounting/tile_crops.py`, so they always see identical pixels.
A box is `mispositioned` if the somite of interest isn't roughly centred
in the *dashed* region.

**Checking progress.** After a session, run either of these to see the
per-annotator / per-severity / per-box-quality breakdown:

```bash
# Django ORM — portable across DB backends, prints dicts
python manage.py shell -c "from django.db.models import Count; from well_mapping.models import SomiteAnnotation; [print(r) for r in SomiteAnnotation.objects.values('annotator', 'box_quality', 'severity').annotate(n=Count('id')).order_by('annotator', 'box_quality', 'severity')]"

# Raw SQL — same numbers, faster to scan
python manage.py dbshell <<< "SELECT annotator, box_quality, severity, COUNT(*) FROM well_mapping_somiteannotation GROUP BY annotator, box_quality, severity ORDER BY annotator, box_quality, severity;"
```

Use the `box_quality` counts to decide whether to build the BoxEditTool:
if `mispositioned` is >10% of your annotations, fixing boxes by hand is
worth ~150 lines of dashboard code; if it's <5%, the flag alone is enough
and the training script will just drop those rows.

Why a DB table and not a JSON file: multiple annotators can rate the same
somite for inter-rater agreement, the training script can pull a clean SQL
join, and there's no file-locking hazard if someone runs the extractor
mid-annotation. Migration is `well_mapping/0037_somiteannotation_*.py` —
run `python manage.py migrate well_mapping` after pulling.

**Step 4 — Check quality** with `annotation_stats`:

```bash
python manage.py annotation_stats
python manage.py annotation_stats --annotator clement --experiment VAST_2026-05
```

Prints per-annotator counts (severity / box_quality / lr_offset), the
class-balance readout for training viability (`OK ≥500`, `ok ≥100`,
`!! <100` per class), inter-rater agreement on overlapping somites, and
per-experiment progress (annotated / available). Run after every
session to know when each class has enough samples and whether two
annotators agree on the same fish.

**Step 5 — Export** the labelled tiles when you're ready to train:

```bash
# Default: class-folder layout (torchvision.datasets.ImageFolder-ready)
python manage.py export_somite_training_set --output_dir data/training_v1

# Restrict to one annotator (clement) for a single-rater baseline
python manage.py export_somite_training_set --annotator clement

# Flat layout — single tiles/ folder, splits via the manifest
python manage.py export_somite_training_set --flat
```

Output layout (default):

```text
data/training_v1/
├── tiles/
│   ├── 0_healthy/   *.png
│   ├── 1_mild/      *.png
│   ├── 2_moderate/  *.png
│   └── 3_severe/    *.png
└── manifest.csv      # tile_path, severity, lr_offset, annotator,
                     # experiment, plate, well, dest_well_id, somite_index
```

Re-crops each somite via `somiteCounting/tile_crops.py` so the exported
PNG is byte-identical to what the annotator labelled in the dashboard.
By default drops `box_quality != 'single'`, `severity IS NULL`, and
`lr_offset=True` — pass `--include_lr_offset` / `--include_unsure` if
you need those for a different training setup.

Still to build:

- **Classifier training** — a `somiteCounting/training_severity.py` that
  reads `manifest.csv` (or `ImageFolder` straight off the class-folder
  layout) and trains a small CNN. Writes `checkpoints/severity_best.pth`
  + test metrics. The export already pre-loads the data; the training
  script just needs the standard ResNet18-transfer pattern from the
  other classifiers.
- **Plumb it into `profile_v1`** — replace the neighbour-comparison
  `_classify_severity` heuristic with a forward pass over each somite's
  tile through the trained checkpoint.

## Roadmap

Planned, in order:

1. ~~**Multi-row predictions schema**~~ — done (see above).
2. **SAM segmentation dashboard** — new Bokeh page where the user clicks
   somites to seed point prompts; MedSAM produces per-somite masks; an
   "auto-segment" button runs grid-prompt + NMS for batch processing.
   Predictions written with `model_name='sam_v1'` and per-somite results
   stored in `per_somite_data`.
3. **Cross-model comparison UI** — the prediction-vs-annotation table and
   the drug plot both gain dropdowns to select which `model_name` to compare
   against the manual annotations. Scatter plots can overlay multiple models
   for the same wells.
4. **Per-somite as a dedicated model** — promote `per_somite_data` JSON to
   a `SomiteInstance` table once cross-well queries on individual somites
   become routine.

---

## Repository contacts

- **Maintainer**: clement.helsens@epfl.ch (UPOATES lab, EPFL)
- **Issue tracker**: GitHub repo issues
