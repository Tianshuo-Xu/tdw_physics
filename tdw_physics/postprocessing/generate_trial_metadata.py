import os, sys, subprocess
from pkgutil import iter_modules
import importlib
import h5py
import numpy as np

from tdw_physics.postprocessing.labels import get_labels_from
import tdw_physics.target_controllers as controllers

def list_controllers():
    cs = []
    for c in iter_modules(controllers.__path__):
        cs.append(c.name)
    return cs

def get_controller_label_funcs_by_class(cls: str = 'MultiDominoes'):
    

if __name__ == '__main__':
    print(list_controllers())
