"""Quality + progress report for the SomiteAnnotation table.

Run after every annotation session to track:

  * per-annotator throughput
  * severity / box_quality / lr_offset breakdowns
  * inter-rater agreement (where 2+ annotators have rated the same somite)
  * class balance for the future severity classifier
  * per-experiment completeness

Examples
--------
    python manage.py annotation_stats
    python manage.py annotation_stats --annotator clement
    python manage.py annotation_stats --experiment VAST_2026-05
"""

from collections import defaultdict

from django.core.management.base import BaseCommand
from django.db.models import Count


SEV_LABELS = {0: 'healthy', 1: 'mild', 2: 'moderate', 3: 'severe', None: 'unsure'}


class Command(BaseCommand):
    help = "Quality + progress report for SomiteAnnotation."

    def add_arguments(self, parser):
        parser.add_argument("--annotator", default=None,
                            help="Limit to a single annotator.")
        parser.add_argument("--experiment", default=None,
                            help="Substring filter on experiment name.")

    def handle(self, *args, **opts):
        from well_mapping.models import (
            SomiteAnnotation, DestWellPropertiesPredicted,
        )

        qs = SomiteAnnotation.objects.all()
        if opts["annotator"]:
            qs = qs.filter(annotator=opts["annotator"])
        if opts["experiment"]:
            qs = qs.filter(
                dest_well__well_plate__experiment__name__icontains=opts["experiment"])

        n_total = qs.count()
        if n_total == 0:
            self.stdout.write(self.style.WARNING(
                "No annotations match these filters."))
            return

        self.stdout.write("=" * 72)
        self.stdout.write(self.style.NOTICE(
            f"SomiteAnnotation stats — {n_total} rows"))
        self.stdout.write("=" * 72)

        # ---- Per-annotator breakdown ----
        self.stdout.write("\nPer-annotator counts:")
        annotators = sorted(qs.values_list('annotator', flat=True).distinct())
        for annot in annotators:
            aq = qs.filter(annotator=annot)
            self.stdout.write(f"\n  [{annot}] {aq.count()} total")

            # Severity (within box_quality='single' — the rest are NULL)
            single = aq.filter(box_quality='single')
            sev_rows = (single.values('severity')
                              .annotate(n=Count('id'))
                              .order_by('severity'))
            self.stdout.write("    severity (single only):")
            for row in sev_rows:
                self.stdout.write(
                    f"      {SEV_LABELS.get(row['severity'], '?'):<10}"
                    f"  {row['n']}")

            # Box quality
            bq_rows = (aq.values('box_quality')
                         .annotate(n=Count('id'))
                         .order_by('-n'))
            self.stdout.write("    box_quality:")
            for row in bq_rows:
                self.stdout.write(
                    f"      {row['box_quality']:<14}{row['n']}")

            lr = aq.filter(lr_offset=True).count()
            self.stdout.write(f"    lr_offset=True: {lr}")

        # ---- Training-set viability ----
        self.stdout.write("\n" + "-" * 72)
        self.stdout.write("Training set viability "
                          "(box_quality='single', severity NOT NULL, lr_offset=False):")
        trainable = qs.filter(box_quality='single',
                              severity__isnull=False,
                              lr_offset=False)
        for sev in (0, 1, 2, 3):
            n = trainable.filter(severity=sev).count()
            # Crude thresholds: <100 too few to train, 100–500 workable
            # baseline, 500+ decent.
            mark = ('OK ' if n >= 500
                    else 'ok ' if n >= 100
                    else '!! ')
            self.stdout.write(f"    [{mark}] {SEV_LABELS[sev]:<10} {n}")
        self.stdout.write(
            "  Legend: OK ≥500 (decent), ok ≥100 (workable baseline), "
            "!! <100 (need more)")

        # ---- Inter-rater agreement ----
        # (dest_well, somite_index) pairs rated by 2+ distinct annotators.
        multi = (qs.values('dest_well_id', 'somite_index')
                   .annotate(n_raters=Count('annotator', distinct=True))
                   .filter(n_raters__gte=2))
        n_overlap = multi.count()
        if n_overlap > 0:
            self.stdout.write("\n" + "-" * 72)
            self.stdout.write(
                f"Inter-rater overlap: {n_overlap} somites rated by 2+ annotators")
            # Pull the actual ratings for overlapping pairs and compute
            # exact-agreement rate on severity.
            overlap_keys = {(m['dest_well_id'], m['somite_index']) for m in multi}
            ratings = defaultdict(list)
            for row in qs.filter(
                dest_well_id__in=[k[0] for k in overlap_keys],
            ).values('dest_well_id', 'somite_index', 'severity'):
                key = (row['dest_well_id'], row['somite_index'])
                if key in overlap_keys:
                    ratings[key].append(row['severity'])
            agree = sum(1 for v in ratings.values() if len(set(v)) == 1)
            pct = 100 * agree / max(1, len(ratings))
            self.stdout.write(
                f"  Exact severity agreement: {agree}/{len(ratings)} "
                f"({pct:.0f}%)")

        # ---- Per-experiment completeness ----
        self.stdout.write("\n" + "-" * 72)
        self.stdout.write("Per-experiment annotation progress "
                          "(somites annotated / somites available in profile_v1):")
        exp_names = sorted(qs.values_list(
            'dest_well__well_plate__experiment__name', flat=True).distinct())
        for exp in exp_names:
            preds = DestWellPropertiesPredicted.objects.filter(
                model_name='profile_v1',
                per_somite_data__isnull=False,
                dest_well__well_plate__experiment__name=exp,
            )
            n_available = sum(
                len((p.per_somite_data or {}).get('somites') or [])
                for p in preds.iterator(chunk_size=200))
            n_done = qs.filter(
                dest_well__well_plate__experiment__name=exp).count()
            pct = (100 * n_done / n_available) if n_available else 0
            self.stdout.write(
                f"  {exp:<40} {n_done:>5} / {n_available:>5}  ({pct:5.1f}%)")
