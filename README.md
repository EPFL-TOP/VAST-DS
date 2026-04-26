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

> **Order matters**: retrain the **orientation classifier first**, then re-run
> orientation correction on every well, then retrain the somite counter and
> validity classifier. The somite/validity training data is typically sourced
> from each well's `corrected_orientation/` subfolder, so refreshing those
> folders with the new orientation model produces cleaner training data.

```bash
# 1. Orientation classifier (BF images)
python -m somiteCounting.training_orientation \
    --input_data_path D:\vast\VAST-DS\training_data \
    --model_save_path checkpoints \
    --epochs 40 --batch_size 8

# 2. Refresh corrected_orientation/ folders for every well using the new model.
#    Walks each experiment, picks the best of {no flip, hflip, vflip, hflip+vflip}
#    on the BF image, and writes the flipped copies to <well>/corrected_orientation/.
python somiteCounting/orientfish.py

# 3. Re-split annotations into train / valid / test and rebuild the on-disk
#    training folders from the freshly canonicalised images. Also sets the
#    use_for_training / use_for_validation / use_for_test flags on
#    DestWellProperties so the Statistics page matches the on-disk split.
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
```

Each script also runs as `python somiteCounting/<file>.py …` if you prefer
the old style — both work.

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

## Roadmap

Planned, in order:

1. **Multi-row predictions schema** — change `DestWellPropertiesPredicted`
   from a OneToOne to a multi-row table keyed by
   `(dest_well, model_name, model_version)` so multiple models (current
   ResNet, future SAM-based segmenter, multi-task net, etc.) can store
   predictions for the same well side by side. Stats and drug-plot pages
   gain a "which model?" selector.
2. **Per-somite data** — initially as a JSONField on the prediction row
   (centroid, area, anterior-posterior position, defective flag per somite),
   promoted to a dedicated `SomiteInstance` model when cross-well queries
   require it.
3. **SAM segmentation dashboard** — new Bokeh page where the user clicks
   somites to seed point prompts; MedSAM produces per-somite masks; an
   "auto-segment" button runs grid-prompt + NMS for batch processing.
   Predictions written to the multi-row schema with `model_name='sam_v1'`.
4. **Cross-model comparison UI** — the prediction-vs-annotation table and
   the drug plot both gain dropdowns to select which model_name to compare
   against the manual annotations. Scatter plots can overlay multiple models
   for the same wells.

---

## Repository contacts

- **Maintainer**: clement.helsens@epfl.ch (UPOATES lab, EPFL)
- **Issue tracker**: GitHub repo issues
