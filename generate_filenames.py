
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

csv_filename = "dump/dominoes/physionpp-dominoes.csv"
local_stem = "dump/dominoes"

dirnames = [d.split('/')[-1] for d in glob(local_stem+'/*')] #arg_names
data_dirs = [os.path.join(local_stem,d) for d in dirnames]

#dataset_name = '{}_{}'.format(bucket_name, stim_version)
stimulus_extension = "mp4" #what's the file extension for the stims? Provide without dot

#print("data_dirs", data_dirs)

## get a list of paths to each one
full_stim_paths,filenames, _  = list_files(data_dirs,stimulus_extension)
#full_map_paths, mapnames, _ = list_files(data_dirs, ext = 'png') #generate filenames and stimpaths for target/zone map
#full_hdf5_paths,hdf5_names, _ = list_files(data_dirs,ext = 'hdf5', exclude=["temp.hdf5"])
full_map_paths = [x.replace("_img.mp4", "_map.png") for x in full_stim_paths]
mapnames = [x.replace("_img.mp4", "_map.png") for x in filenames]
full_hdf5_paths = [x.replace("_img.mp4", ".hdf5") for x in full_stim_paths]
hdf5_names = [x.replace("_img.mp4", ".hdf5") for x in filenames]

# assert the files exists
for map_path in full_map_paths + full_hdf5_paths:
	assert(os.path.exists(map_path))

#print(full_stim_paths)

with open(csv_filename, 'w', newline="") as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(["full_stim_paths", 'filenames', 'full_map_paths', 'mapnames', 'full_hdf5_paths', 'hdf5_names'])
    for fp, fn, fm, mn, fh, hn in zip(full_stim_paths, filenames, full_map_paths, mapnames, full_hdf5_paths,hdf5_names):
        csvwriter.writerow([fp, fn, fm, mn, fh, hn])

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







