# calculate scenarios red-hit-yello length statistics
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json

path_root = "/mnt/fs4/hsiaoyut/tdw_physics/data/"
#path_root = "/media/htung/Extreme SSD/fish/tdw_physics/data/"

nframes_list_cross_scenario = []
scenario_names = ["dominoes", "containment", "collision", "drop", "linking", "rollingSliding", "towers", "clothSagging"]


scenario_names = ["clothSagging"]



for scenario_name in scenario_names:
    print("processing", scenario_name)
    modes = ["train", "valid", "train_readout", "valid_readout"]

    scene_root = os.path.join(path_root, scenario_name)
    arg_names = [arg for arg in os.listdir(os.path.join(path_root, scenario_name)) if arg is not "tfrecords"]


    nframes_list = []
    total_count = 0
    total_contact_count = 0
    for arg_name in arg_names:
        print("        -", arg_name)

        for mode in modes:

            arg_full_path = os.path.join(scene_root, arg_name, mode)

            hdf5names = [f for f in os.listdir(arg_full_path) if f.endswith(".hdf5")]
            ndata = len(hdf5names)

            for trial_id in range(0, ndata):
                total_count += 1
                filename = os.path.join(arg_full_path, f"{trial_id:04}.hdf5")
                #print("processing", filename)
                f = h5py.File(filename, "r")

                nframes = len(f["frames"])

                contact = False
                for frame_id in range(nframes):
                    if f["frames"][f"{frame_id:04}"]["labels"]["target_contacting_zone"][()]:
                        contact = True
                        break
                if contact:
                    nframes_list.append(frame_id)
                    total_contact_count += 1


    plt.figure()
    maxi = np.max(nframes_list)
    mini = np.min(nframes_list)
    plt.hist(nframes_list, bins=20)
    plt.title(f"histogram of trial red-hit-yellow length for {scenario_name} \n (#hit_trials/#trials: {total_contact_count}/{total_count}, max: {maxi}, min: {mini})")
    plt.ylabel("count")
    plt.xlabel("trial length")
    #plt.show()
    plt.savefig(f"vis2/RHYlenght_{scenario_name}.png")
    nframes_list_cross_scenario += nframes_list


# plt.figure()
# maxi = np.max(nframes_list_cross_scenario)
# mini = np.min(nframes_list_cross_scenario)
# plt.hist(nframes_list_cross_scenario, bins=20)
# plt.title(f"histogram of trial red-hit-yellow length  for all scenarios \n (#trials: {len(nframes_list_cross_scenario)}, max: {maxi}, min: {mini})")
# plt.ylabel("count")
# plt.xlabel("trial length")
# #plt.show()
# plt.savefig(f"vis/RHYlenght_all.png")

import ipdb; ipdb.set_trace()