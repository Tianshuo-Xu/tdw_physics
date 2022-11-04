
import os
from glob import glob
import numpy as np
import h5py
import csv
import pickle
from datetime import date

def list_files(paths, ext='mp4', exclude=[]):
    """Pass list of folders if there are stimuli in multiple folders.
    Make sure that the containing folder is informative, as the rest of the path is ignored in naming.
    Also returns filenames as uploaded to S3"""
    if type(paths) is not list:
        paths = [paths]
    results = []
    names = []
    for path in paths:
        filenames = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.%s' % ext))]
        filenames = [y for y in filenames if y.split("/")[-1] not in exclude]
        results += filenames
        #import ipdb; ipdb.set_trace()

        names += [os.path.basename(os.path.dirname(y))+'_'+os.path.split(y)[1] for y in filenames]

        #names += [os.path.basename(os.path.dirname(y))+'_'+os.path.split(y)[1] for x in os.walk(path) for y in glob(os.path.join(x[0], '*.%s' % ext)) if y not in exclude]
    hdf5s = [r.split("_img.")[0]+".hdf5" for r in results]
    return results,names,hdf5s

##### inputs
phy_name = "mass"
set_id = 0 #set_id
copy_id = 0


phy_to_scenario = dict()
phy_to_scenario["mass"] = ["mass_dominoes", "mass_collision"]#, "mass_waterpush"]
phy_to_scenario["bouncy"] = ["bouncy_platform", "bouncy_wall"]
phy_to_scenario["friction"] = ["friction_platform", "friction_cloth", "friction_collision"]
phy_to_scenario["deform"] = ["deform_clothhang", "deform_clothhit", "deform_clothdrop"]
phy_to_scenario["viscosity"] = ["viscosity_fluidslope", "viscosity_fluiddrop"]


with_dv2 = False

data_root = "/media/htung/Extreme SSD/fish/tdw_physics/dump_tmp"
data_ori = "/media/htung/Extreme SSD/fish/tdw_physics/dump_mini4"
date_str = date.today().strftime("%y%m%d")
csv_filename = f"physionpp-{phy_name}_merge_{date_str}.csv"

if with_dv2:
    csv_filename = csv_filename.replace(".csv", "_dv2.csv")

csv_filename = os.path.join(data_root, csv_filename)
data_dirs = []
for scene_name in phy_to_scenario[phy_name]:
    local_stem = os.path.join(data_root, f"{scene_name}_pp-copy{copy_id}")
    tmp_mp4_folder = local_stem + "_mp4"
    dirnames = [d.split('/')[-1] for d in glob(local_stem+'/*')] #arg_names
    data_dirs += [os.path.join(local_stem,d) for d in dirnames]


#dataset_name = '{}_{}'.format(bucket_name, stim_version)
stimulus_extension = "mp4" #what's the file extension for the stims? Provide without dot

#print("data_dirs", data_dirs)

## get a list of paths to each one
full_mp4_paths, mp4filenames, _  = list_files(data_dirs,stimulus_extension)
candidate_full_stim_paths = [p for p in full_mp4_paths if "_img.mp4" in p]

full_stim_paths = []
full_stim_apaths = []
if with_dv2:
    full_stim_ifpaths = []
filenames = []
filenames_a = []
filenames_if = []
target_hit_zone_labels = []
start_frame_after_curtain = []
start_frame_for_prediction = []

star_phy = []

for idx, full_stim_path in enumerate(candidate_full_stim_paths):


    pkl_filename = full_stim_path.replace(data_root, data_ori).replace("_img.mp4", ".pkl")
    with open(pkl_filename, "rb") as f:
        data = pickle.load(f)

    # take label from the last interaction
    mass_list = data['static']['mass']
    object_ids = data['static']['object_ids'].tolist()
    star_mass = data['static']['star_mass']
    star_phy.append(star_mass)

    target_hit_zone_labels.append(data['static']['does_target_contact_zone'])
    start_frame_for_prediction.append(data['static']['start_frame_for_prediction'])

    filename = "_".join(full_stim_path.split("/")[-2:])

    #print(full_stim_path, n_interactions, )
    full_stim_paths.append(full_stim_path)
    full_stim_apaths.append(full_stim_path.replace("_img.mp4", "_aimg.mp4"))
    if with_dv2:
        full_stim_ifpaths.append(full_stim_path.replace("_img.mp4", "_ifimg.mp4"))
    filenames.append(filename)
    filenames_a.append(filename.replace("_img.mp4", "_aimg.mp4"))

    #[x.replace("_img.mp4", f"_{n_interactions-1:03}.hdf5") for x in filenames]

#filenames = ["_".join(p.split("/")[-2:]) for p in full_stim_paths]
#full_map_paths, mapnames, _ = list_files(data_dirs, ext = 'png') #generate filenames and stimpaths for target/zone map
#full_hdf5_paths,hdf5_names, _ = list_files(data_dirs,ext = 'hdf5', exclude=["temp.hdf5"])
# use map at the first frame, and hdf5 at the last frame
full_map_paths = [x.replace("_img.mp4", "_000_map.png") for x in full_stim_paths]
mapnames = [x.replace("_img.mp4", "_000_map.png") for x in filenames]

print("writing file to ...", csv_filename)
with open(csv_filename, 'w', newline="") as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')

    if with_dv2:
        csvwriter.writerow(["full_stim_paths", "full_stim_apaths", "full_stim_ifpaths", 'filenames', 'filenames_a', 'filenames_if', 'full_map_paths', 'mapnames', 'full_hdf5_paths', 'hdf5_names', 'target_hit_zone_labels', 'start_frame_after_curtain', 'star_phy', 'nonstar_phy'])
        for fp, fap, fifp, fn, fna, fnif, fm, mn, fh, hn, lb, sf, sp, nsp  in zip(full_stim_paths, full_stim_apaths, full_stim_ifpaths, filenames, filenames_a, filenames_if, full_map_paths, mapnames, full_hdf5_paths,hdf5_names, target_hit_zone_labels, start_frame_after_curtain, star_phy, nonstar_phy):
            csvwriter.writerow([fp.replace(local_stem, tmp_mp4_folder), fap.replace(local_stem, tmp_mp4_folder),\
                                fifp.replace(local_stem, tmp_mp4_folder),\
                                 fn, fna, fnif, fm, mn, fh, hn, lb, sf, sp, nsp])
    else:
        csvwriter.writerow(["full_stim_paths", "full_stim_apaths", 'filenames', 'filenames_a', 'target_hit_zone_labels', 'start_frame_for_prediction'])
        for fp, fap, fn, fna, lb, sf  in zip(full_stim_paths, full_stim_apaths, filenames, filenames_a, target_hit_zone_labels, start_frame_for_prediction):
            csvwriter.writerow([fp.replace(local_stem, tmp_mp4_folder), fap.replace(local_stem, tmp_mp4_folder),\
                                 fn, fna, lb, sf])

print("red_hits_yellow ratio",np.sum(target_hit_zone_labels), "/", len(target_hit_zone_labels))
assert(len(filenames) == len(mapnames))
print("=====")
#print(filenames)
print('We have {} stimuli to evaluate.'.format(len(full_stim_paths)))


# read red_hits_yellow labels from hdf5
# for hdf5_path in full_hdf5_paths:
#     try:

#         import ipdb; ipdb.set_trace()
#         hdf5 = h5py.File(hdf5_path,'r') #get the static part of the HDF5
#         stim_name = str(np.array(hdf5['static']['stimulus_name']))
#         metadatum = {} #metadata for the current stimulus
#         for key in hdf5['static'].keys():
#             datum = np.array(hdf5['static'][key])
#             if datum.shape == (): datum = datum.item() #unwrap non-arrays
#             metadatum[key] = datum
#         import ipdb; ipdb.set_trace()
#         #close file
#         hdf5.close()
#         metadata[name] = metadatum
#     except Exception as e:
#         print("Error with",hdf5_path,":",e)
#         continue












