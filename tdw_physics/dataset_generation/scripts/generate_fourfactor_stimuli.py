import time
import os, sys
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import json
import random

import tdw_physics.target_controllers.relations as rel
from tdw_physics.dataset_generation.configs.fourfactor_experimental import \
    (set_a, set_b, set_a_params, set_b_params,
     contain_params, occlude_params, collide_params, miss_params,
     common_params)

SAVE_FRAME = 90

def build_scenarios():

    scenarios = []

    return scenarios

def main(args):

    ## build the scenarios
    pass

if __name__ == '__main__':

    args = rel.get_relational_args("relational")

    import platform, os
    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":0." + str(args.gpu)
        else:
            os.environ["DISPLAY"] = ":0"

    main(args)
