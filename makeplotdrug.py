import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "VAST_DS.settings")  # <-- change this
django.setup()

from well_mapping.models import DestWellPropertiesPredicted

derivation_name = "LY411575 Stock1"
experiment_names = ["VAST_2026-01-26", "VAST_2026-02-02"]


dest=DestWellPropertiesPredicted.objects.filter(
        valid=True,
        dest_well__source_well__drugs__derivation_name=derivation_name,
        dest_well__well_plate__experiment__name__in=experiment_names,
    ).distinct()

for d in dest:
    #print('-------------- ',len(d))
    print('-------------- ',d)

#qs = (
#    DestWellPropertiesPredicted.objects
#    .filter(
#        valid=True,
#        dest_well__source_well__drugs__derivation_name=derivation_name,
#        dest_well__well_plate__experiment__name__in=experiment_names,
#    )
#    .values("n_total_somites", "n_bad_somites")
#)