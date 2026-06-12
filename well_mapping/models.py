from django.db import models
from django.urls import reverse
from django.db.models.signals import pre_delete, post_delete
from django.dispatch import receiver

WELLPLATETYPES = (
    ('6', '6 well plate'),
    ('12', '12 well plate'),
    ('24', '24 well plate'),
    ('48', '48 well plate'),
    ('96', '96 well plate'),
    ('384', '384 well plate'),
    ('1536', '1536 well plate'),
    ('supp', 'supplementary well plate'),
    ('other', 'other well plate')
)

WELLPLATEROW = (
    ('A', 'A'), 
    ('B', 'B'),
    ('C', 'C'),
    ('D', 'D'),
    ('E', 'E'),
    ('F', 'G'),
    ('G', 'F'),
    ('H', 'H'),
    ('Z', 'Z'),
)

WELLPLATECOL = (
    ('1', '1'),
    ('2', '2'),
    ('3', '3'),
    ('4', '4'),
    ('5', '5'),
    ('6', '6'),
    ('7', '7'),
    ('8', '8'),
    ('9', '9'),
    ('10', '10'),
    ('11', '11'),
    ('12', '12'),
    ('999', '999'),
)


#___________________________________________________________________________________________
class Experiment(models.Model):
    name            = models.CharField(max_length=200, help_text="name of the experiment.")
    date            = models.DateField(blank=True, null=True, help_text="Date of the experiment")
    description     = models.TextField(blank=True, max_length=2000, help_text="Description of the experiment")
    pyrat_id        = models.CharField(max_length=200, help_text="pyrat ID of the experiment", default='', blank=True, null=True)

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "{0}, {1}".format(self.name, self.date)

    class Meta:
        ordering = ["name"]

#___________________________________________________________________________________________
class SourceWellPlate(models.Model):
    plate_type      = models.CharField(max_length=200, help_text="Type of the source well plate (e.g. 24, 48, etc.)", choices=WELLPLATETYPES)
    experiment      = models.OneToOneField(Experiment, default='', on_delete=models.CASCADE, related_name='source_plate')
    n_well_supp     = models.IntegerField(default=0, help_text="Number of supplementary wells in the source well plate", blank=True, null=True)

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "exp={0}, date={1}, plate_type={2}, n_well_supp={3}".format(self.experiment.name, self.experiment.date, self.plate_type, self.n_well_supp)

    class Meta:
        ordering = ["experiment__name"]

#___________________________________________________________________________________________
class DestWellPlate(models.Model):
    plate_type      = models.CharField(max_length=200, help_text="Type of the destination well plate (e.g. 48, 96, etc.)", choices=WELLPLATETYPES)
    plate_number    = models.IntegerField(default=1, help_text="Number of the destination well plate in the experiment (e.g. 1, 2, etc.)")
    experiment      = models.ForeignKey(Experiment, default='', on_delete=models.CASCADE, related_name='dest_plate')

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "exp={0}, n_plate={1}, date={2}, plate_type={3}".format(self.experiment.name, self.plate_number, self.experiment.date, self.plate_type)

    class Meta:
        ordering = ["experiment__name"]

#___________________________________________________________________________________________
class SourceWellPosition(models.Model):
    position_col  = models.CharField(max_length=10, choices=WELLPLATECOL, help_text="Column position on VAST the well plate", default='ZZZ')
    position_row  = models.CharField(max_length=10, choices=WELLPLATEROW, help_text="Row position on VAST the well plate", default='999')
    well_plate    = models.ForeignKey(SourceWellPlate,  default='', on_delete=models.CASCADE)
    is_supp       = models.BooleanField(default=False, help_text="Is this a supplementary well position?", blank=True)
    valid         = models.BooleanField(default=True, help_text="can be imaged with VAST flag", blank=True, null=True)
    comments      = models.TextField(blank=True, max_length=2000, help_text="Comments if any", null=True)

    #drug          = models.ManyToManyField(Drug, default='', related_name='drugs', blank=True, null=True, help_text="Source well positions of the drug in the source well plate")  

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "exp={0}, pos={1}{2}, is_supp={3}".format(self.well_plate.experiment.name, self.position_row, self.position_col, self.is_supp)

    def remove_drug(self, drug):
            """
            Remove the link to a drug and delete the drug if it's not linked anywhere else.
            Also clears DestWellPosition links if needed.
            """
            DestWellPosition.objects.filter(source_well=self).update(source_well=None) 

            # Step 1: Remove relation
            self.drugs.remove(drug)

            # Step 2: If drug is orphaned, delete it and cleanup DestWellPosition
            if drug.position.count() == 0:
                self._cleanup_dest_wells_for_drug(drug)
                drug.delete()

    def delete(self, *args, **kwargs):
        """
        When deleting a SourceWellPosition, clean up related drugs and DestWellPositions.
        """
        for drug in list(self.drugs.all()):
            self.drugs.remove(drug)
            if drug.position.count() == 0:
                self._cleanup_dest_wells_for_drug(drug)
                drug.delete()

        super().delete(*args, **kwargs)

    def _cleanup_dest_wells_for_drug(self, drug):
        """
        For each SourceWellPosition linked to the drug, clear the source_well field
        in DestWellPositions pointing to them.
        """
        from .models import DestWellPosition  # avoid circular import
        for swp in drug.position.all():
            DestWellPosition.objects.filter(source_well=swp).update(source_well='')


    def unmap_dest_wells(self):
        """
        Unmap this SourceWellPosition from all DestWellPositions that point to it.
        """
        updated_count = DestWellPosition.objects.filter(source_well=self).update(source_well=None)
        return updated_count  # number of rows updated


#___________________________________________________________________________________________
class DestWellPosition(models.Model):
    position_col    = models.CharField(max_length=10, choices=WELLPLATECOL, help_text="Column position on VAST the well plate", default='ZZZ')
    position_row    = models.CharField(max_length=10, choices=WELLPLATEROW, help_text="Row position on VAST the well plate", default='999')
    well_plate      = models.ForeignKey(DestWellPlate,  default='', on_delete=models.CASCADE)
    source_well     = models.ForeignKey(SourceWellPosition, default='', on_delete=models.SET_DEFAULT, blank=True, null=True)

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "exp={0}, pos={1}{2}, n_plate={3}".format(self.well_plate.experiment.name, self.position_row, self.position_col, self.well_plate.plate_number)

#___________________________________________________________________________________________
class Drug(models.Model):
    slims_id        = models.CharField(max_length=200, help_text="slims ID of the drug derivation.")
    derivation_name = models.CharField(max_length=200, help_text="name of the drug derivation.", default='', blank=True)
    concentration   = models.FloatField(help_text="Concentration of the drug derivation (mMol/L) or Percentage of the drug derivation (%).", default=-9999, blank=True, null=True)
    #valid           = models.BooleanField(default=True, help_text="can be imaged with VAST flag", blank=True)
    #drug_derivation = models.ForeignKey(SlimsDrugDerivation,  default='', on_delete=models.CASCADE, blank=True, null=True)
    position        = models.ManyToManyField(SourceWellPosition, default='', related_name='drugs', blank=True, help_text="Source well positions of the drug in the source well plate")  
    
    def __str__(self):
       
        exp_names = {
            pos.well_plate.experiment.name
            for pos in self.position.all()
        }
        # turn that set into a comma‑separated string
        exp_list = ", ".join(sorted(exp_names)) if exp_names else "(no experiment)"
        return (
            f"derivation_name={self.derivation_name} "
            f"slims_id={self.slims_id} "
            f"concentration={self.concentration} "
            f"experiment={exp_list}"
        )

# 1) Before a position is deleted, stash its drugs on the instance
@receiver(pre_delete, sender=SourceWellPosition)
def _stash_drugs_for_cleanup(sender, instance, **kwargs):
    # pull them off the DB so we have a Python list to work with later
    instance._drugs_to_check = list(instance.drugs.all())

# 2) After the position (and its join‐table rows) is gone, clean up orphans
@receiver(post_delete, sender=SourceWellPosition)
def _cleanup_orphan_drugs(sender, instance, **kwargs):
    for drug in getattr(instance, '_drugs_to_check', []):
        # if this was the drug’s last position, we can delete it
        if not drug.position.exists():
            drug.delete()


#___________________________________________________________________________________________
class DestWellProperties(models.Model):
    dest_well           = models.OneToOneField(DestWellPosition, default='', on_delete=models.CASCADE, related_name='dest_well_properties')
    n_total_somites     = models.IntegerField(default=-9999, help_text="Number of total somites in this well", blank=True, null=True)
    n_bad_somites       = models.IntegerField(default=-9999, help_text="Number of bad somites in this well", blank=True, null=True)
    n_total_somites_err = models.IntegerField(default=0, help_text="Number of total somites error", blank=True, null=True)
    n_bad_somites_err   = models.IntegerField(default=0, help_text="Number of bad somites error", blank=True, null=True)    
    comments            = models.TextField(blank=True, max_length=2000, help_text="Comments if any", null=True)
    valid               = models.BooleanField(default=True, help_text="is a valid fish image", blank=True, null=True)
    correct_orientation = models.BooleanField(default=True, help_text="is the fish correctly oriented (head to the left)?", blank=True, null=True)
    use_for_training    = models.BooleanField(default=False, help_text="should be used for training", blank=True, null=True)
    use_for_validation  = models.BooleanField(default=False, help_text="should be used for validation", blank=True, null=True)
    use_for_test        = models.BooleanField(default=False, help_text="held out for the final test report (never seen during training)", blank=True, null=True)

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "exp={0}, pos={1}{2}, n_plate={3}, n_total_somites={4}, n_bad_somites={5}, valid={6}".format(self.dest_well.well_plate.experiment.name, self.dest_well.position_row, self.dest_well.position_col, self.dest_well.well_plate.plate_number, self.n_total_somites, self.n_bad_somites, self.valid)
    
#___________________________________________________________________________________________
class DestWellPropertiesPredicted(models.Model):
    """Predictions produced by a DL model for a destination well.

    One row per `(dest_well, model_name, model_version)` triple — so multiple
    models can store predictions for the same well side-by-side and be
    compared. Use `latest_prediction(dest, model_name=...)` to fetch the
    most recent prediction for a given well/model.
    """
    dest_well           = models.ForeignKey(DestWellPosition, on_delete=models.CASCADE,
                                            related_name='predictions')
    model_name          = models.CharField(max_length=200, default='resnet_v1', db_index=True,
                                           help_text="Identifier of the model that produced this prediction "
                                                     "(e.g. 'resnet_v1', 'sam_v1', 'multitask_v1')")
    model_version       = models.CharField(max_length=200, default='', blank=True,
                                           help_text="Optional version/checkpoint tag (git hash, date, etc.)")
    predicted_at        = models.DateTimeField(auto_now=True,
                                               help_text="Last time this prediction was written")

    n_total_somites     = models.IntegerField(default=-9999, help_text="Number of total somites in this well", blank=True, null=True)
    n_bad_somites       = models.IntegerField(default=-9999, help_text="Number of bad somites in this well", blank=True, null=True)
    valid               = models.BooleanField(default=True, help_text="is a valid fish image", blank=True, null=True)
    correct_orientation = models.BooleanField(default=True, help_text="is the fish correctly oriented (head to the left)?", blank=True, null=True)

    # Per-somite data, e.g. from SAM segmentation:
    #   [{"index": 0, "centroid_x": 320, "centroid_y": 110, "area": 850,
    #     "ap_position": 0.42, "severity": 1, "comments": "..."}, ...]
    per_somite_data     = models.JSONField(blank=True, null=True, default=None,
                                           help_text="Per-somite metrics from segmentation models (list of dicts)")

    class Meta:
        unique_together = (('dest_well', 'model_name', 'model_version'),)
        indexes = [
            models.Index(fields=['dest_well', 'model_name']),
            models.Index(fields=['model_name', '-predicted_at']),
        ]

    def __str__(self):
        return ("exp={0}, pos={1}{2}, n_plate={3}, model={4}, n_total_somites={5}, "
                "n_bad_somites={6}, valid={7}").format(
            self.dest_well.well_plate.experiment.name,
            self.dest_well.position_row, self.dest_well.position_col,
            self.dest_well.well_plate.plate_number, self.model_name,
            self.n_total_somites, self.n_bad_somites, self.valid)


class SomiteAnnotation(models.Model):
    """Human label of a single somite's defect severity.

    One row per ``(dest_well, somite_index, annotator)`` — multiple
    annotators can rate the same somite for inter-rater agreement.
    ``somite_index`` matches the ``index`` field inside
    ``DestWellPropertiesPredicted.per_somite_data['somites']`` for the
    well's ``profile_v1`` row, which is the spatial AP order along the
    fish. ``severity=None`` means the annotator marked it 'unsure' (the row
    still exists so we don't keep re-showing it).
    """
    SEVERITY_CHOICES = (
        (0, 'healthy'),
        (1, 'mild'),
        (2, 'moderate'),
        (3, 'severe'),
    )
    # Whether the bbox cleanly contains one somite. Training filters to
    # 'single'; everything else is recorded so we can later (a) audit the
    # detector's failure modes, (b) feed the bad boxes into a box-quality
    # head if we ever want one.
    BOX_QUALITY_CHOICES = (
        ('single',        'single somite, well-centred'),
        ('multiple',      'multiple somites in the box'),
        ('empty',         'no somite — false positive from detector'),
        ('mispositioned', 'somite is partly out of the box / off-centre'),
    )

    dest_well     = models.ForeignKey(DestWellPosition, on_delete=models.CASCADE,
                                      related_name='somite_annotations')
    somite_index  = models.IntegerField(
        help_text="Matches per_somite_data['somites'][i]['index'] for the well's profile_v1 row.")
    severity      = models.IntegerField(
        choices=SEVERITY_CHOICES, null=True, blank=True,
        help_text="Annotator's rating. NULL = 'unsure' OR box_quality != 'single' "
                  "(severity is undefined when the box doesn't contain exactly one somite).")
    box_quality   = models.CharField(
        max_length=20, choices=BOX_QUALITY_CHOICES, default='single',
        help_text="Quality of the detector's bounding box. Only 'single' rows "
                  "are usable for severity-classifier training.")
    annotator     = models.CharField(max_length=120, default='unknown', db_index=True)
    annotated_at  = models.DateTimeField(auto_now=True)
    notes         = models.TextField(blank=True, default='')

    class Meta:
        unique_together = (('dest_well', 'somite_index', 'annotator'),)
        indexes = [
            models.Index(fields=['dest_well', 'somite_index']),
            models.Index(fields=['annotator', '-annotated_at']),
        ]

    def __str__(self):
        sev = 'unsure' if self.severity is None else dict(self.SEVERITY_CHOICES)[self.severity]
        return (f"{self.dest_well} somite#{self.somite_index} "
                f"= {sev} (by {self.annotator})")


def latest_prediction(dest_well, model_name='resnet_v1', model_version=None):
    """Return the most recent DestWellPropertiesPredicted row for a given
    well/model, or None. Optional model_version constrains further.

    Centralised here so callers don't all have to know the multi-row schema.
    """
    qs = DestWellPropertiesPredicted.objects.filter(
        dest_well=dest_well, model_name=model_name)
    if model_version is not None:
        qs = qs.filter(model_version=model_version)
    return qs.order_by('-predicted_at').first()
