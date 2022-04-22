import os, sys, json, h5py, copy
import numpy as np
import random
from typing import List, Dict, Tuple

## tdw
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelRecord, MaterialLibrarian
from tdw.output_data import OutputData, Transforms, Images, CameraMatrices

## tdw_physics
import tdw_physics.rigidbodies_dataset as rigid
import tdw_physics.util as utils
from tdw_physics.postprocessing.labels import get_all_label_funcs

MODEL_LIBRARIES = utils.MODEL_LIBRARIES

class OcpRigidDataset(rigid.RigidbodiesDataset):
    """
    Abstract class that places an agent ("target") and patient ("zone")
    object, along with intermediate structure that gets overwritten by 
    child classes.
    """
    DEFAULT_RAMPS = [r for r in MODEL_LIBRARIES['models_full.json'].records
                     if 'ramp_with_platform_30' in r.name]
    CUBE = [r for r in MODEL_LIBRARIES['models_flex.json'].records
            if 'cube' in r.name][0]
    PRINT = False

    def __init__(self,
                 port: int = None,
                 room: str = 'box',
                 **kwargs
                 ):
        super().__init__(port=port, **kwargs)

        
                 
    


