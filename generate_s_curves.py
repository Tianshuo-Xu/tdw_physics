import os
import glob
import h5py
import math
import numpy as np

folder = "/media/htung/Extreme SSD/fish/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_3"
folder="/home/hsiaoyut/2021/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_0"


target_varname = "star_mass"

target_params = []
labels = []

for hdf5_file in glob.glob(folder + "/*_001.hdf5"):
	print(hdf5_file)

	f = h5py.File(hdf5_file, "r")

	target_params.append(f["static"][target_varname][()])
	labels.append(float(f["static"]["does_target_contact_zone"][()]))

print(target_params)


target_params = [math.log10(param/2) for param in target_params]
import matplotlib.pyplot as plt
#scatter plot
for oid, param in enumerate(target_params):
    if param > 0.5 and labels[oid] < 0.2:
        print("heavy object with negative outcome", oid)

plt.scatter(target_params, labels)
plt.xlabel("log(mass)")
plt.ylabel("red hits yellow")


nbins = 10
n, _ = np.histogram(target_params, bins=nbins)
sy, _ = np.histogram(target_params, bins=nbins, weights=labels)
sy2, _ = np.histogram(target_params, bins=nbins, weights=labels)
mean = sy / n
std = np.sqrt(sy2/n - mean*mean)
plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='r-')
#hist plot

plt.show()
plt.savefig("s_curve.png")

print(len(labels))

#print(target_params)
