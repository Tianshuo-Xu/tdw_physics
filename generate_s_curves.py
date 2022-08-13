import os
import socket
import glob
import h5py
import math
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

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

if "ccncluster" in socket.gethostname():
    data_root = "/mnt/fs4/hsiaoyut/physion++/analysis"
else:
    data_root = "/media/htung/Extreme SSD/fish/tdw_physics/dump"

sname = "mass_dominoes_pp"
folder = os.path.join(data_root, sname)
#import ipdb; ipdb.set_trace()
filenames = os.listdir(folder)
restrict = "pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_curtain" #"pilot_it2_drop_simple_box" #"bouncy_platform-use_blocker_with_hole=1" #"target_cone-tscale_0.35,0.5,0.35"
remove = "" #"simple_box1"
filenames = [filename for filename in filenames if restrict in filename]

target_varname = "star_mass" #"star_mass" #"star_mass", "star_deform"
merge_by =  "all" #"all""tscale"
#merge_by = "tscale"

set_dict = collections.defaultdict(list)

for filename in filenames:
    info_dict = split_info(filename)
    if merge_by == "all":
        set_dict["all"].append(filename)
    elif merge_by:
        set_dict[info_dict[merge_by]].append(f/home/hsiaoyut/2021/tdw_physics/home/hsiaoyut/2021/tdw_physics/home/hsiaoyut/2021/tdw_physics/home/hsiaoyut/2021/tdw_physicsilename)
    else:
        set_dict[filename].append(filename)
import ipdb; ipdb.set_trace()

for set_id, merge_var_name in enumerate(set_dict):
    target_params = []
    labels = []

    for filename in tqdm(set_dict[merge_var_name]):
        if remove and remove in filename:
            continue


        for pkl_file in glob.glob(os.path.join(folder, filename) + "/*.pkl"):
            print(pkl_file)
            with open(pkl_file, "rb") as f:
                f = pickle.load(f)

            #print(f['static']['cloth_material'])
            target_params.append(f["static"][target_varname])
            labels.append(float(f["static"]["does_target_contact_zone"]))

        print(labels)
        print(target_params)
    """

    for hdf5_file in glob.glob(folder + "/*_001.hdf5"):
    	print(hdf5_file)

    	f = h5py.File(hdf5_file, "r")

    	target_params.append(f["static"][target_varname][()])
    	labels.append(float(f["static"]["does_target_contact_zone"][()]))
    """

    if target_varname in ["star_mass"]:
        target_params = [math.log10(param) for param in target_params]
    else:
        target_params = [param for param in target_params]

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
    ax = plt.gca()
    ax.set_ylim([-0.1, 1.5])


    nbins = 8
    n, _ = np.histogram(target_params, bins=nbins)
    sy, _ = np.histogram(target_params, bins=nbins, weights=labels)
    sy2, _ = np.histogram(target_params, bins=nbins, weights=labels)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='-', label=merge_var_name)

    #hist plot
#plt.legend(loc="lower left")
plt.legend(loc="upper left")
plt.show()
plt.savefig(f"s_curve_{sname}.png")

print(len(labels))

#print(target_params)
