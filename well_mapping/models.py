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
class Images(models.Model):
    files  = models.JSONField(help_text="images associated to a drug", default=dict)
    well_position   = models.ForeignKey(DestWellPosition,  default='', on_delete=models.SET_DEFAULT)


#___________________________________________________________________________________________
class Scores(models.Model):
    images = models.ForeignKey(Images, default='', on_delete=models.SET_DEFAULT)
    algo   = models.JSONField(help_text="algorithm parameters, version", default=dict)
    scores = models.JSONField(help_text="scores associated to this algo", default=dict)






#___________________________________________________________________________________________
#class DrugDerivationWellPlate(models.Model):

    #experiment      = models.ForeignKey(Experiment, default='', on_delete=models.CASCADE)

#    def __str__(self):
#        """String for representing the Model object (in Admin site etc.)"""
#        return "{0}, {1}".format(self.experiment.name, self.experiment.date)

    #class Meta:
    #    ordering = ["experiment__name"]

#___________________________________________________________________________________________
#class DrugDerivationWellCluster(models.Model):

#    slims_id        = models.CharField(max_length=200, help_text="slims ID of the drug derivation.")
#    concentration   = models.CharField(max_length=200, help_text="Concentration of the drug derivation (mMol/L).")
    #well_plate      = models.ForeignKey(DrugDerivationWellPlate, default='', on_delete=models.SET_DEFAULT)
#    comments        = models.TextField(blank=True, max_length=2000, help_text="Comments if any")

    #def __str__(self):
    #    """String for representing the Model object (in Admin site etc.)"""
    #    return "id={0}, concentration={1}".format(self.slims_id, self.concentration)
    
#___________________________________________________________________________________________
#class DrugDerivationWellPosition(models.Model):
#    ROW = (
#        ('A', 'A'), 
#        ('B', 'B'),
#        ('C', 'C'),
#        ('D', 'D'),
#        ('E', 'E'),
#        ('F', 'F'),
#    )
#    COL = (
#        ('1', '1'),
#        ('2', '2'),
#        ('3', '3'),
#        ('4', '4'),
#        ('5', '5'),
#        ('6', '6'),
#        ('7', '7'),
#        ('8', '8'),
#    )
    #24 well plate  48 position_col    = #1->8 position_row    = #A->F

#    position_col    = models.CharField(max_length=10, choices=ROW, help_text="Column position on the small well plate", default='A')
#    position_row    = models.CharField(max_length=10, choices=ROW, help_text="Row position in the small well plate", default='1')
    #drug_derivation = models.ForeignKey(SlimsDrugDerivation,  default='', blank=True, null=True, on_delete=models.SET_DEFAULT)
    #experiment      = models.ForeignKey(Experiment, default='', on_delete=models.SET_DEFAULT)
    #cluster         = models.ForeignKey(DrugDerivationWellCluster, default='', on_delete=models.SET_DEFAULT)

#    def __str__(self):
#        """String for representing the Model object (in Admin site etc.)"""
#        drug=None
#        if self.drug_derivation!=None:
#            drug=self.drug_derivation.drug
        #return "exp name={0}, drug={1}, pos col={2}, pos row={3}".format(self.cluster.well_plate.experiment.name, drug, self.position_col, self.position_row)


#___________________________________________________________________________________________
#class VASTWellPlate(models.Model):
    #experiment  = models.ForeignKey(Experiment, default='', on_delete=models.CASCADE)

#    def __str__(self):
#        """String for representing the Model object (in Admin site etc.)"""
#        return "{0}, {1}".format(self.experiment.name, self.experiment.date)

    #class Meta:
     #   ordering = ["experiment__name"]


#___________________________________________________________________________________________
#class VASTWellCluster(models.Model):
    #plate  = models.ForeignKey(VASTWellPlate, default='', on_delete=models.SET_DEFAULT)

#___________________________________________________________________________________________
#class VASTWellPosition(models.Model):
#    ROW = (
#        ('A', 'A'), 
#        ('B', 'B'),
#        ('C', 'C'),
#        ('D', 'D'),
#        ('E', 'E'),
#        ('F', 'G'),
#        ('G', 'F'),
#        ('H', 'H'),
#    )
#    COL = (
#        ('1', '1'),
#        ('2', '2'),
#        ('3', '3'),
#        ('4', '4'),
#        ('5', '5'),
#        ('6', '6'),
#        ('7', '7'),
#        ('8', '8'),
#        ('9', '9'),
#        ('10', '10'),
#        ('11', '11'),
#        ('12', '12'),
#    )
#    position_col    = models.CharField(max_length=10, choices=ROW, help_text="Column position on VAST the well plate", default='A')
#    position_row    = models.CharField(max_length=10, choices=COL, help_text="Row position on VAST the well plate", default='1')
    #drug_derivation = models.ForeignKey(DrugDerivationWellPosition,  default='', on_delete=models.SET_DEFAULT)
    #cluster         = models.ForeignKey(VASTWellCluster,  default='', on_delete=models.SET_DEFAULT)


    #def __str__(self):
    #    """String for representing the Model object (in Admin site etc.)"""
    #    return "col={0}, row={1}, {2}".format(self.position_col, self.position_row, self.drug_derivation)




#___________________________________________________________________________________________
#class SlimsDrug(models.Model):
#class SlimsDrugPowder(models.Model):
#    name        = models.CharField(max_length=200, help_text="name of the drug.", default='', blank=True, null=True)
#    cas_mumber  = models.CharField(max_length=200, help_text="cas number of the drug.", default='', blank=True, null=True)
#    supplier    = models.CharField(max_length=200, help_text="supplier of the drug.", default='', blank=True, null=True)
#    reference   = models.CharField(max_length=200, help_text="reference number of the drug.", default='', blank=True, null=True)
#    #storage     = models.CharField(max_length=200, help_text="storage of the drug.", default='', blank=True, null=True)
#    #quantity    = models.FloatField(help_text="Drug quantity (mg).", default=0.0, blank=True, null=True)
#    #hazard      = models.CharField(max_length=200, help_text="hazard of the drug.", default='', blank=True, null=True)
#    slims_id    = models.CharField(max_length=200, help_text="slims ID of the drug.", default='', blank=True, null=True)
#
#    def __str__(self):
#        """String for representing the Model object (in Admin site etc.)"""
#        return "{0}, {1}".format(self.name, self.supplier)


#___________________________________________________________________________________________
#class SlimsDrugDerivation(models.Model):
#class SlimsDrugStock(models.Model):
#    batch_number          = models.CharField(max_length=200, help_text="batch number of the drug derivation.", default='', blank=True, null=True)
#    powder_diluent        = models.CharField(max_length=200, help_text="powder diluent of the drug derivation.", default='', blank=True, null=True)
#    stock_concentration   = models.CharField(max_length=200, help_text="stock concentration of the drug derivation.", default='', blank=True, null=True)
#    #ultrasonic            = models.BooleanField(help_text="Ultrasonic treatment of the drug derivation.", blank=True, null=True)
#    #storage_freezer       = models.CharField(max_length=200, help_text="Storage freezer of the drug derivation.", default='', blank=True, null=True)
#    #storage_position      = models.CharField(max_length=200, help_text="Storage position of the drug derivation.", default='', blank=True, null=True)
#    slims_id              = models.CharField(max_length=200, help_text="slims ID of the drug derivation.", default='', blank=True, null=True)
#    name                  = models.CharField(max_length=200, help_text="name of the slim drug derivation.", default='', blank=True, null=True)
#    #slim_drug             = models.ForeignKey(SlimsDrug,  default='', on_delete=models.CASCADE)
#    slim_drug             = models.ForeignKey(SlimsDrugPowder,  default='', on_delete=models.CASCADE)
#
#    def __str__(self):
#        """String for representing the Model object (in Admin site etc.)"""
#        return "{0}, {1}, {2}".format(self.slim_drug.name, self.batch_number, self.stock_concentration)