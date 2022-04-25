from argparse import ArgumentParser
import sys
import subprocess
import h5py
import json
import shlex
import copy
import importlib
from PIL import Image
import io
import os
import math
from tqdm import tqdm
import numpy as np
from pathlib import Path
from enum import Enum
import random
from typing import List, Dict, Tuple
from collections import OrderedDict
from weighted_collection import WeightedCollection
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelRecord, MaterialLibrarian
from tdw.output_data import OutputData, Transforms, Images, CameraMatrices
from tdw_physics.rigidbodies_dataset import (RigidbodiesDataset,
                                             get_random_xyz_transform,
                                             get_range,
                                             handle_random_transform_args)
from tdw_physics.util import (MODEL_LIBRARIES, FLEX_MODELS, MODEL_CATEGORIES,
                              MATERIAL_TYPES, MATERIAL_NAMES,
                              get_parser,
                              xyz_to_arr, arr_to_xyz, str_to_xyz,
                              none_or_str, none_or_int, int_or_bool)
from tdw_physics.postprocessing.stimuli import pngs_to_mp4
from tdw_physics.postprocessing.labels import (get_all_label_funcs,
                                               get_labels_from)
from tdw_physics.util_geom import save_obj
PRIMITIVE_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records if not r.do_not_use]
FULL_NAMES = [r.name for r in MODEL_LIBRARIES['models_full.json'].records if not r.do_not_use]
PASSES = ["_img", "_depth", "_normals", "_flow", "_id"]
ZONE_COLOR = [255,255,0]
TARGET_COLOR = [255,0,0]


### add new scenario base here


def get_class_name(query):
    if query == "dominoes":
        from tdw_physics.target_controllers.dominoes_base import DominoesScenario
        return DominoesScenario
    else:
        raise ValueError

if __name__ == "__main__":
    import platform, os

    # python
    scenarios = sys.argv[1].split("=")
    assert(scenarios[0] == "--scenario_types"), f"please run your commend line as: python --scenario_types=\"dominoes,dominoes\" @args... "
    scenarios = scenarios[1].split(",")
    assert(len(scenarios) == 2)

    scenario_main = scenarios[0]
    scenario_second = scenarios[1]


    #import ipdb; ipdb.set_trace()
    secondscene_fn=get_class_name(scenario_second)
    mainscene = get_class_name(scenario_main)(secondscene_fn=secondscene_fn)


    args = mainscene.args

    if bool(args.run):
        mainscene.controller.run(num=args.num,
                 output_dir=args.dir,
                 temp_path=args.temp,
                 width=args.width,
                 height=args.height,
                 framerate=args.framerate,
                 write_passes=args.write_passes.split(','),
                 save_passes=args.save_passes.split(','),
                 save_movies=args.save_movies,
                 save_labels=args.save_labels,
                 save_meshes=args.save_meshes,
                 args_dict=vars(args))
    else:
        end = DomC.communicate({"$type": "terminate"})
        print([OutputData.get_data_type_id(r) for r in end])
