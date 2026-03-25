import os
import django
import matplotlib.pyplot as plt
import numpy as np
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

n_total_notv = []
n_bad_notv = []
for d in dest:

    n_total.append(d.n_total_somites)
    n_bad.append(d.n_bad_somites)

for d in dest_notvalid:
    n_total_notv.append(d.n_total_somites)
    n_bad_notv.append(d.n_bad_somites)


print('mean tot   = ',np.mean(np.array(n_total)))
print('median tot = ',np.median(np.array(n_total)))
print('std tot    = ',np.std(np.array(n_total)))

print('mean bad   = ',np.mean(np.array(n_bad)))
print('median bad = ',np.median(np.array(n_bad)))
print('std bad    = ',np.std(np.array(n_bad)))

plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.hist(n_total, bins=10, alpha=0.6, label="Total somites and valid fish={0}".format(len(n_total)), density=True)
plt.legend()
plt.title(f"Distribution of somites for {derivation_name}")
plt.xlabel("Number of somites")
plt.ylabel("Frequency")

plt.text(23, 0.5, r'mean tot   = {:.1f}'.format(np.mean(np.array(n_total))), fontsize=15)
plt.text(23, 0.4, r'median tot = {:.1f}'.format(np.median(np.array(n_total))), fontsize=15)
plt.text(23, 0.3, r'std tot    = {:.1f}'.format(np.std(np.array(n_total))), fontsize=15)

plt.subplot(212)
plt.hist(n_bad, bins=20, alpha=0.6, label="Defective somites", density=True)
plt.xlabel("Number of somites")
plt.ylabel("Frequency")
plt.text(0, 0.09, r'mean defective   = {:.1f}'.format(np.mean(np.array(n_bad))), fontsize=15)
plt.text(0, 0.075, r'median defective = {:.1f}'.format(np.median(np.array(n_bad))), fontsize=15)
plt.text(0, 0.06, r'std defective    = {:.1f}'.format(np.std(np.array(n_bad))), fontsize=15)

plt.legend()
plt.show()


plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.hist(n_total_notv, bins=20, alpha=0.6, label="Total somites", density=True)
plt.legend()
plt.title(f"Distribution of somites for {derivation_name} and not valid fish={0}".format(len(n_bad_notv)))
plt.xlabel("Number of somites")
plt.ylabel("Frequency")
plt.text(5, 0.5, r'mean tot   = {:.1f}'.format(np.mean(np.array(n_total_notv))), fontsize=15)
plt.text(5, 0.4, r'median tot = {:.1f}'.format(np.median(np.array(n_total_notv))), fontsize=15)
plt.text(5, 0.3, r'std tot    = {:.1f}'.format(np.std(np.array(n_total_notv))), fontsize=15)

plt.subplot(212)
plt.hist(n_bad_notv, bins=20, alpha=0.6, label="Defective somites", density=True)
plt.xlabel("Number of somites")
plt.ylabel("Frequency")
plt.text(0, 0.09, r'mean defective   = {:.1f}'.format(np.mean(np.array(n_bad_notv))), fontsize=15)
plt.text(0, 0.075, r'median defective = {:.1f}'.format(np.median(np.array(n_bad_notv))), fontsize=15)
plt.text(0, 0.06, r'std defective    = {:.1f}'.format(np.std(np.array(n_bad_notv))), fontsize=15)
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