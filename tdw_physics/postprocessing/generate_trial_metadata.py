import os, sys, glob, subprocess
from pkgutil import iter_modules
import importlib
from pathlib import Path
from collections import OrderedDict
from typing import List,Dict,Tuple
import h5py, json
import numpy as np
import argparse
from tqdm import tqdm

from tdw_physics.postprocessing.labels import *
import tdw_physics.target_controllers as controllers

def list_controllers():
    cs = []
    for c in iter_modules(controllers.__path__):
        cs.append(c.name)
    return cs

def get_controller_label_funcs_by_class(cls: str = 'MultiDominoes'):

    cs = list_controllers()
    Class = None
    for c in cs:
        module = importlib.import_module("tdw_physics.target_controllers." + c)
        if cls in module.__dict__.keys():
            Class = getattr(module, cls)
            funcs = Class.get_controller_label_funcs(cls)
            return funcs



def compute_metadata_from_stimuli(
        stimulus_dir : str,
        file_pattern : str = "*.hdf5",
        controller_class: str = None,
        label_funcs: List[type(list_controllers)] = [],
        add_controller_funcs: bool = True,
        overwrite: bool = False,
        outfile: str = 'metadata') -> None:

    # get the hdf5s in the directory
    stims = sorted(glob.glob(stimulus_dir + file_pattern))

    # try to infer the controller class
    if controller_class is None:
        meta_file = Path(stimulus_dir).joinpath('metadata.json')
        if meta_file.exists():
            trial_meta = json.loads(meta_file.read_text())[0]
            controller_class = str(trial_meta['controller_name'])
        else:
            raise ValueError("Controller classname could not be read from existing metadata.json")

    # add the label funcs
    if add_controller_funcs:
        label_funcs += get_controller_label_funcs_by_class(controller_class)

    # iterate over stims
    metadata = []
    for stimpath in tqdm(stims):
        f = h5py.File(stimpath, 'r')
        trial_meta = OrderedDict()
        trial_meta = get_labels_from(f, label_funcs, res=trial_meta)
        metadata.append(trial_meta)
        f.close()

    # write out new metadata
    json_str = json.dumps(metadata, indent=4)    
    meta_file = Path(stimulus_dir).joinpath(outfile + ('' if overwrite else '_post') + '.json')
    meta_file.write_text(json_str, encoding='utf-8')
    print("Wrote new metadata: %s\nfor %d trials" % (str(meta_file), len(metadata)))
    return

def concatenate_metadata_fields(
        stimulus_dir: str,
        metafile: str = 'metadata.json',
        fields: List[str] = None,        
        outfile: str = 'metadata_by_field.json') -> None:

    meta = Path(stimulus_dir).joinpath(metafile)
    metadata = json.loads(meta.read_text(encoding='utf-8'))

    if fields is None:
        fields = [str(k) for k in metadata[0].keys()]

    outdata = {}
    for field in fields:
        data = [m[field] for m in metadata]
        outdata[field] = data

    json_str = json.dumps(outdata, indent=4)
    outfile = Path(stimulus_dir).joinpath(outfile)
    outfile.write_text(json_str, encoding='utf-8')
        
    return
        
        
        
if __name__ == '__main__':
    # print(get_controller_label_funcs_by_class())

    stim_dir = sys.argv[1]
    compute_metadata_from_stimuli(stim_dir,
                                  label_funcs=[
                                      stimulus_name,
                                      probe_name,
                                      probe_segmentation_color,
                                      static_model_names
                                  ],
                                  add_controller_funcs=False,
                                  overwrite= True,
                                  outfile='model_names')

    concatenate_metadata_fields(stim_dir,
                                metafile='model_names.json',
                                fields=None)
