from django.db import models
from django.urls import reverse

#___________________________________________________________________________________________
class Experiment(models.Model):
    name        = models.CharField(max_length=200, help_text="name of the experiment.")
    date        = models.DateField(blank=True, null=True, help_text="Date of the experiment")
    description = models.TextField(blank=True, max_length=2000, help_text="Description of the experiment")

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "{0}, {1}".format(self.name, self.date)


#___________________________________________________________________________________________
class SlimsDrug(models.Model):
    name        = models.CharField(max_length=200, help_text="name of the drug.")
    cas_mumber  = models.CharField(max_length=200, help_text="cas number of the drug.")
    supplier    = models.CharField(max_length=200, help_text="supplier of the drug.")
    reference   = models.CharField(max_length=200, help_text="reference number of the drug.")
    storage     = models.CharField(max_length=200, help_text="storage of the drug.")
    quantity    = models.CharField(max_length=200, help_text="quantity of the drug.")
    hazard      = models.CharField(max_length=200, help_text="hazard of the drug.")
    slims_id    = models.CharField(max_length=200, help_text="slims ID of the drug.")

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "{0}, {1}".format(self.name, self.supplier)


#___________________________________________________________________________________________
class SlimsDrugDerivation(models.Model):

    batch_number          = models.CharField(max_length=200, help_text="batch number of the drug derivation.")
    powder_diluent        = models.CharField(max_length=200, help_text="powder diluent of the drug derivation.")
    stock_concentration   = models.CharField(max_length=200, help_text="stock concentration of the drug derivation.")
    ultrasonic            = models.BooleanField(default=False, blank=True, help_text="Ultrasonic treatment of the drug derivation.")
    storage_freezer       = models.CharField(max_length=200, help_text="Storage freezer of the drug derivation.")
    storage_position      = models.CharField(max_length=200, help_text="Storage position of the drug derivation.")
    drug                  = models.ForeignKey(SlimsDrug,  default='', on_delete=models.SET_DEFAULT)
    slims_id              = models.CharField(max_length=200, help_text="slims ID of the drug derivation.")

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "{0}, {1}, {2}".format(self.drug.name, self.batch_number, self.stock_concentration)


#___________________________________________________________________________________________
class DrugDerivationWellPlate(models.Model):

    experiment      = models.ForeignKey(Experiment, default='', on_delete=models.SET_DEFAULT)

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "id={0}, name={1}".format(self.id, self.experiment.name)


#___________________________________________________________________________________________
class DrugDerivationWellCluster(models.Model):

    slims_id        = models.CharField(max_length=200, help_text="slims ID of the drug derivation.")
    concentration   = models.CharField(max_length=200, help_text="Concentration of the drug derivation (mMol/L).")
    clusters        = models.ForeignKey(DrugDerivationWellPlate, default='', on_delete=models.SET_DEFAULT)

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "id={0}, concentration={1}".format(self.slims_id, self.concentration)
    
#___________________________________________________________________________________________
class DrugDerivationWellPosition(models.Model):
    ROW = (
        ('A', 'A'), 
        ('B', 'B'),
        ('C', 'C'),
        ('D', 'D'),
        ('E', 'E'),
        ('F', 'F'),
    )
    COL = (
        ('1', '1'),
        ('2', '2'),
        ('3', '3'),
        ('4', '4'),
        ('5', '5'),
        ('6', '6'),
        ('7', '7'),
        ('8', '8'),
    )
    #24 well plate  48 position_col    = #1->8 position_row    = #A->F

    position_col    = models.CharField(max_length=10, choices=ROW, help_text="Column position on the small well plate", default='A')
    position_row    = models.CharField(max_length=10, choices=ROW, help_text="Row position in the small well plate", default='1')
    drug_derivation = models.ForeignKey(SlimsDrugDerivation,  default='', on_delete=models.SET_DEFAULT)
    #experiment      = models.ForeignKey(Experiment, default='', on_delete=models.SET_DEFAULT)
    cluster         = models.ForeignKey(DrugDerivationWellCluster, default='', on_delete=models.SET_DEFAULT)

    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "exp name={0}, drug={1}, pos col={2}, pos row={3}".format(self.experiment.name, self.drug_derivation.drug, self.position_col, self.position_row)


#___________________________________________________________________________________________
class VASTWellPlate(models.Model):
    experiment  = models.ForeignKey(Experiment, default='', on_delete=models.SET_DEFAULT)


#___________________________________________________________________________________________
class VASTWellCluster(models.Model):
    plate  = models.ForeignKey(VASTWellPlate, default='', on_delete=models.SET_DEFAULT)

#___________________________________________________________________________________________
class VASTWellPosition(models.Model):
    ROW = (
        ('A', 'A'), 
        ('B', 'B'),
        ('C', 'C'),
        ('D', 'D'),
        ('E', 'E'),
        ('F', 'G'),
        ('G', 'F'),
        ('H', 'H'),
    )
    COL = (
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
    )
    position_col    = models.CharField(max_length=10, choices=ROW, help_text="Column position on VAST the well plate", default='A')
    position_row    = models.CharField(max_length=10, choices=COL, help_text="Row position on VAST the well plate", default='1')
    drug_derivation = models.ForeignKey(DrugDerivationWellPosition,  default='', on_delete=models.SET_DEFAULT)
    cluster         = models.ForeignKey(VASTWellCluster,  default='', on_delete=models.SET_DEFAULT)


    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return "col={0}, row={1}, {2}".format(self.position_col, self.position_row, self.drug_derivation)


#___________________________________________________________________________________________
class Images(models.Model):
    vast_wellplate = models.ForeignKey(VASTWellPosition,  default='', on_delete=models.SET_DEFAULT)
    files          = models.JSONField(help_text="images associated to a VAST well", default=dict)
    cluster        = models.ForeignKey(VASTWellCluster,  default='', on_delete=models.SET_DEFAULT)


#___________________________________________________________________________________________
class Scores(models.Model):
    images = models.ForeignKey(Images, default='', on_delete=models.SET_DEFAULT)
    algo   = models.JSONField(help_text="algorithm parameters, version", default=dict)
    scores = models.JSONField(help_text="scores associated to this algo", default=dict)