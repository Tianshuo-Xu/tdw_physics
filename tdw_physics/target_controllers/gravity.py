import sys, os
from argparse import ArgumentParser
import h5py
import json
import copy
import importlib
import numpy as np
from enum import Enum
import random
from typing import List, Dict, Tuple
from weighted_collection import WeightedCollection
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelRecord, MaterialLibrarian
from tdw.output_data import OutputData, Transforms
from tdw_physics.rigidbodies_dataset import (RigidbodiesDataset,
                                             get_random_xyz_transform,
                                             get_range,
                                             handle_random_transform_args)
from tdw_physics.util import (MODEL_LIBRARIES,
                              get_parser,
                              xyz_to_arr, arr_to_xyz, str_to_xyz,
                              none_or_str, none_or_int, int_or_bool)

from tdw_physics.target_controllers.dominoes import Dominoes, MultiDominoes, get_args
from tdw_physics.postprocessing.labels import is_trial_valid

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}

def get_gravity_args(dataset_dir: str, parse=True):

    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')

    parser.add_argument("--ramp",
                        type=int,
                        default=1,
                        help="Whether to place the probe object on the top of a ramp")
    parser.add_argument("--probe",
                        type=str,
                        default="sphere",
                        help="comma-separated list of possible probe objects")
    parser.add_argument("--pscale",
                        type=str,
                        default="0.2",
                        help="scale of probe objects")
    parser.add_argument("--pmass",
                        type=str,
                        default="1.0",
                        help="scale of probe objects")    
    parser.add_argument("--collision_axis_length",
                        type=float,
                        default=3.0,
                        help="How far to put the probe and target")

    # camera
    parser.add_argument("--camera_distance",
                        type=float,
                        default=2.75,
                        help="radial distance from camera to centerpoint")
    parser.add_argument("--camera_min_angle",
                        type=float,
                        default=0,
                        help="minimum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_max_angle",
                        type=float,
                        default=90,
                        help="maximum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_min_height",
                        type=float,
                        default=1.5,
                         help="min height of camera")
    parser.add_argument("--camera_max_height",
                        type=float,
                        default=2.5,
                        help="max height of camera")
        


    def postprocess(args):

        args = domino_postproc(args)

        return args

    if not parse:
        return (parser, postprocess)

    args = parser.parse_args()
    args = postprocess(args)

    if args.training_data_mode:
        args.dir = os.path.join(args.dir, 'training_data')
        args.random = 0
        args.seed = args.seed + 1
        args.color = args.pcolor = args.mcolor = args.rcolor = None            
        args.remove_zone = 1
        args.remove_target = 1
        args.save_passes = ""
        args.save_movies = False
        args.tower_cap = MODEL_NAMES

    return args
    
class Gravity(Dominoes):

    DEFAULT_RAMPS = [r for r in MODEL_LIBRARIES['models_full.json'].records if 'ramp_with_platform_30' in r.name]
    
    def __init__(self,
                 port: int = 1071,
                 middle_scale_range=1.0,
                 middle_color=None,
                 middle_material=None,
                 remove_middle=False,
                 **kwargs):

        super().__init__(port=port, **kwargs)

        self._middle_types = self.DEFAULT_RAMPS        
        self.middle_scale_range = middle_scale_range
        self.middle_color = middle_color
        self.middle_material = middle_material
        self.remove_middle = remove_middle

    def _build_intermediate_structure(self) -> List[dict]:

        ramp_pos = TDWUtils.VECTOR3_ZERO
        ramp_rot = TDWUtils.VECTOR3_ZERO

        self.middle = random.choice(self._middle_types)
        self.middle_type = self.middle.name
        self.middle_id = self._get_next_object_id()
        self.middle_scale = get_random_xyz_transform(self.middle_scale_range)
        
        commands = self.add_ramp(
            record = self.middle,
            position = ramp_pos,
            rotation = ramp_rot,
            scale = self.middle_scale,
            o_id = self.middle_id,
            add_data = True)

        # give the ramp a texture and color
        commands.extend(
            self.get_object_material_commands(
                self.middle, self.middle_id, self.get_material_name(self.middle_material)))        
        rgb = self.middle_color or self.random_color(exclude=self.target_color)
        commands.append(
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": self.middle_id})            

        return commands

if __name__ == '__main__':

    args = get_gravity_args("gravity")

    GC = Gravity(

        # gravity specific
        middle_scale_range=args.mscale,
        
        # domino specific
        target_zone=args.zone,
        zone_location=args.zlocation,
        zone_scale_range=args.zscale,
        zone_color=args.zcolor,
        zone_friction=args.zfriction,        
        target_objects=args.target,
        probe_objects=args.probe,
        target_scale_range=args.tscale,
        target_rotation_range=args.trot,
        probe_scale_range=args.pscale,
        probe_mass_range=args.pmass,
        target_color=args.color,
        probe_color=args.pcolor,
        middle_color=args.mcolor,
        collision_axis_length=args.collision_axis_length,
        force_scale_range=args.fscale,
        force_angle_range=args.frot,
        force_offset=args.foffset,
        force_offset_jitter=args.fjitter,
        force_wait=args.fwait,
        remove_target=bool(args.remove_target),
        remove_zone=bool(args.remove_zone),
        
        ## not scenario-specific
        room=args.room,
        randomize=args.random,
        seed=args.seed,
        camera_radius=args.camera_distance,
        camera_min_angle=args.camera_min_angle,
        camera_max_angle=args.camera_max_angle,
        camera_min_height=args.camera_min_height,
        camera_max_height=args.camera_max_height,
        monochrome=args.monochrome,
        material_types=args.material_types,
        target_material=args.tmaterial,
        probe_material=args.pmaterial,
        middle_material=args.mmaterial,
        zone_material=args.zmaterial,
        distractor_types=args.distractor,
        distractor_categories=args.distractor_categories,
        num_distractors=args.num_distractors,
        occluder_types=args.occluder,
        occluder_categories=args.occluder_categories,
        num_occluders=args.num_occluders,
        occlusion_scale=args.occlusion_scale,
        remove_middle=args.remove_middle,
        use_ramp=bool(args.ramp),
        ramp_color=args.rcolor
    )


    if bool(args.run):
        GC.run(num=args.num,
               output_dir=args.dir,
               temp_path=args.temp,
               width=args.width,
               height=args.height,
               write_passes=args.write_passes.split(','),               
               save_passes=args.save_passes.split(','),
               save_movies=args.save_movies,
               save_labels=args.save_labels,               
               args_dict=vars(args)
        )
    else:
        GC.communicate({"$type": "terminate"})    
