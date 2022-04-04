
import os
from glob import glob
import numpy as np
import h5py
import csv

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

csv_filename = "dump/dominoes/physionpp-dominoes_merge.csv"
local_stem = "dump/dominoes"
tmp_mp4_folder = "dump/dominoes_mp4"
n_interactions = 2
dirnames = [d.split('/')[-1] for d in glob(local_stem+'/*')] #arg_names
data_dirs = [os.path.join(local_stem,d) for d in dirnames]

#dataset_name = '{}_{}'.format(bucket_name, stim_version)
stimulus_extension = "mp4" #what's the file extension for the stims? Provide without dot

#print("data_dirs", data_dirs)

## get a list of paths to each one
full_mp4_paths,mp4filenames, _  = list_files(data_dirs,stimulus_extension)
candidate_full_stim_paths = [p.replace("_000_img", "_img") for p in full_mp4_paths if "_000_img" in p]


full_hdf5_paths = []
hdf5_names = []
full_stim_paths = []
full_stim_apaths = []
filenames = []
filenames_a = []
target_hit_zone_labels = []
start_frame_after_curtain = []
for idx, full_stim_path in enumerate(candidate_full_stim_paths):
    file_prefix = full_stim_path[:-7]
    n_inters = len([p for p in full_mp4_paths if p.startswith(file_prefix)])
    if n_inters != n_interactions:
        continue
    print(full_stim_path, n_interactions)

    # take label from the last interaction
    full_hdf5_paths.append(full_stim_path.replace("_img.mp4", f"_{n_interactions-1:03}.hdf5"))
    hdf5 = h5py.File(full_hdf5_paths[-1],'r') #get the static part of the HDF5
    accumulate_frame = 0
    for inter_id in range(n_interactions - 1):
        full_stim_path_prev = full_stim_path.replace("_img.mp4", f"_{inter_id:03}.hdf5")
        hdf5_prev = h5py.File(full_stim_path_prev,'r')
        accumulate_frame += len(hdf5_prev["frames"])

    accumulate_frame += hdf5['static']['start_frame_after_curtain'][()]
    target_hit_zone_labels.append(hdf5['static']['does_target_contact_zone'][()])
    start_frame_after_curtain.append(accumulate_frame)
    filename = "_".join(full_stim_path.split("/")[-2:])

    #print(full_stim_path, n_interactions, )
    full_stim_paths.append(full_stim_path)
    full_stim_apaths.append(full_stim_path.replace("_img.mp4", "_aimg.mp4"))
    filenames.append(filename)
    filenames_a.append(filename.replace("_img.mp4", "_aimg.mp4"))
    hdf5_names.append(filename.replace("_img.mp4", f"_{n_interactions-1:03}.hdf5"))
    #[x.replace("_img.mp4", f"_{n_interactions-1:03}.hdf5") for x in filenames]

#filenames = ["_".join(p.split("/")[-2:]) for p in full_stim_paths]
#full_map_paths, mapnames, _ = list_files(data_dirs, ext = 'png') #generate filenames and stimpaths for target/zone map
#full_hdf5_paths,hdf5_names, _ = list_files(data_dirs,ext = 'hdf5', exclude=["temp.hdf5"])
# use map at the first frame, and hdf5 at the last frame
full_map_paths = [x.replace("_img.mp4", "_000_map.png") for x in full_stim_paths]
mapnames = [x.replace("_img.mp4", "_000_map.png") for x in filenames]


# assert the files exists
for map_path in full_map_paths + full_hdf5_paths:
	assert(os.path.exists(map_path)), map_path

#print(full_stim_paths)

with open(csv_filename, 'w', newline="") as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(["full_stim_paths", "full_stim_apaths", 'filenames', 'filenames_a', 'full_map_paths', 'mapnames', 'full_hdf5_paths', 'hdf5_names', 'target_hit_zone_labels', 'start_frame_after_curtain'])
    for fp, fap, fn, fna, fm, mn, fh, hn, lb, sf  in zip(full_stim_paths, full_stim_apaths, filenames, filenames_a, full_map_paths, mapnames, full_hdf5_paths,hdf5_names, target_hit_zone_labels, start_frame_after_curtain):
        csvwriter.writerow([fp.replace(local_stem, tmp_mp4_folder), fap.replace(local_stem, tmp_mp4_folder),\
                             fn, fna, fm, mn, fh, hn, lb, sf])


print("red_hits_yellow ratio",np.sum(target_hit_zone_labels), "/", len(target_hit_zone_labels))
assert(len(filenames) == len(mapnames))
assert(len(hdf5_names) == len(filenames))
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







