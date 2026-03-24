import os
import django
import matplotlib.pyplot as plt

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
dest_notvalid=DestWellPropertiesPredicted.objects.filter(
        valid=False,
        dest_well__source_well__drugs__derivation_name=derivation_name,
        dest_well__well_plate__experiment__name__in=experiment_names,
    ).distinct()
print('n valid ',len(dest))
print('n not valid ',len(dest_notvalid))

n_total = []
n_bad = []
for d in dest:

    n_total.append(d.n_total_somites)
    n_bad.append(d.n_bad_somites)



plt.figure(figsize=(10, 5))

# Histogram for total somites
plt.hist(n_total, bins=25, alpha=0.6, label="Total somites")

# Histogram for bad somites
plt.hist(n_bad, bins=25, alpha=0.6, label="Bad somites")

plt.xlabel("Number of somites")
plt.ylabel("Frequency")
plt.title(f"Distribution of somites for {derivation_name}")
plt.legend()
plt.show()

#qs = (
#    DestWellPropertiesPredicted.objects
#    .filter(
#        valid=True,
#        dest_well__source_well__drugs__derivation_name=derivation_name,
#        dest_well__well_plate__experiment__name__in=experiment_names,
#    )
#    .values("n_total_somites", "n_bad_somites")
#)