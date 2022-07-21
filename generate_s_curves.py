import os
import glob
import h5py
import math
import numpy as np
import pickle
import collections

#folder = "/media/htung/Extreme SSD/fish/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_3"
#folder="/home/hsiaoyut/2021/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_0"

def split_info(filename):
    infos = filename.split("-")[1:]
    info_dict = dict()
    for info in infos:
        arg_name, arg_value = info.split("=")
        info_dict[arg_name] = arg_value
    return info_dict
folder = "/media/htung/Extreme SSD/fish/tdw_physics/dump/bouncy_platform_pp"
filenames = os.listdir(folder)
restrict = "" #"target_bowl" #"target_cone-tscale_0.35,0.5,0.35"
filenames = [filename for filename in filenames if restrict in filename]

target_varname = "star_bouncy" #"star_mass"
#merge_by = "target"
merge_by = "tscale"

set_dict = collections.defaultdict(list)

for filename in filenames:
    info_dict = split_info(filename)
    if merge_by:
        set_dict[info_dict[merge_by]].append(filename)
    else:
        set_dict[filename].append(filename)

for set_id, merge_var_name in enumerate(set_dict):
    target_params = []
    labels = []

    for filename in set_dict[merge_var_name]:
        for pkl_file in glob.glob(os.path.join(folder, filename) + "/*.pkl"):
            print(pkl_file)
            with open(pkl_file, "rb") as f:
                f = pickle.load(f)
            target_params.append(f["static"][target_varname])
            labels.append(float(f["static"]["does_target_contact_zone"]))
    """

    for hdf5_file in glob.glob(folder + "/*_001.hdf5"):
    	print(hdf5_file)

    	f = h5py.File(hdf5_file, "r")

    	target_params.append(f["static"][target_varname][()])
    	labels.append(float(f["static"]["does_target_contact_zone"][()]))
    """
    print(target_params)

    if target_varname in ["star_mass"]:
        target_params = [math.log10(param) for param in target_params]
    else:
        target_params = [param for param in target_params]
    import matplotlib.pyplot as plt
    #scatter plot
    for oid, param in enumerate(target_params):
        if param > 0.5 and labels[oid] < 0.2:
            print("heavy object with negative outcome", oid)

    plt.scatter(target_params, labels)
    if target_varname in ["star_mass"]:
        plt.xlabel(f"log({target_varname})")
    else:
        plt.xlabel(f"{target_varname}")
    plt.ylabel("red hits yellow")


    nbins = 10
    n, _ = np.histogram(target_params, bins=nbins)
    sy, _ = np.histogram(target_params, bins=nbins, weights=labels)
    sy2, _ = np.histogram(target_params, bins=nbins, weights=labels)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='-', label=merge_var_name)

    #hist plot
plt.legend(loc="lower left")
plt.show()
plt.savefig("s_curve.png")

print(len(labels))

#print(target_params)
