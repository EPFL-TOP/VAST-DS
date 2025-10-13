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
    valid           = models.BooleanField(default=True, help_text="can be imaged with VAST flag", blank=True)
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
    valid               = models.BooleanField(default=True, help_text="should be used for training", blank=True, null=True)

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "exp={0}, pos={1}{2}, n_plate={3}, n_good_somites={4}, n_bad_somites={5}, valid={6}".format(self.dest_well.well_plate.experiment.name, self.dest_well.position_row, self.dest_well.position_col, self.dest_well.well_plate.plate_number, self.n_total_somites, self.n_bad_somites, self.valid)