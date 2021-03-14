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

from tdw_physics.target_controllers.dominoes import Dominoes, MultiDominoes
from tdw_physics.target_controllers.towers import Tower, get_tower_args
from tdw_physics.postprocessing.labels import is_trial_valid

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}


def get_linking_args(dataset_dir: str, parse=True):
    """
    Combine Tower-specific args with general Dominoes args
    """
    common = get_parser(dataset_dir, get_help=False)
    tower, tower_postproc = get_tower_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, tower], conflict_handler='resolve', fromfile_prefix_chars='@')

    parser.add_argument("--middle",
                        type=none_or_str,
                        default='torus',
                        help="Which type of object to use as the links")
    parser.add_argument("--mscale",
                        type=none_or_str,
                        default="0.4,0.2,0.4",
                        help="The xyz scale ranges for each link object")
    parser.add_argument("--num_middle_objects",
                        type=int,
                        default=1,
                        help="How many links to use")

    parser.add_argument("--ramp",
                        type=none_or_int,
                        default=1,
                        help="Whether to place the probe object on the top of a ramp")    

    # for generating training data without zones, targets, caps, and at lower resolution
    parser.add_argument("--training_data_mode",
                        action="store_true",
                        help="Overwrite some parameters to generate training data without target objects, zones, etc.")

    def postprocess(args):

        # parent postprocess
        args = tower_postproc(args)

        return args


    if not parse:
        return (parser, postprocess)

    args = parser.parse_args()
    args = postprocess(args)
    
    # produce training data
    if args.training_data_mode:
        args.dir = os.path.join(args.dir, 'training_data')
        args.random = 0
        args.seed = args.seed + 1
        args.color = args.pcolor = args.mcolor = args.rcolor = None            
        args.remove_zone = 1
        args.remove_target = 1
        args.save_passes = ""
        args.save_movies = False

    return args

class Linking(Tower):
    
    def __init__(self,
                 port: int = 1071,
                 
                 # stand base
                 base_object=None,
                 base_scale_range=0.5,
                 
                 # what object the links attach to
                 attachment_object='cylinder',
                 attachment_scale_range={'x': 0.2, 'y': 0.5, 'z': 0.2},
                 attachment_fixed_to_base=False,
                 
                 # what the links are, how many, and which is the target
                 link_objects='torus',
                 link_scale_range=0.5,
                 num_links=1,
                 target_link_idx=None,

                 # generic
                 use_ramp=True,
                 **kwargs):

        super().__init__(port=port, tower_cap=[], **kwargs)

        self.use_ramp = use_ramp
        self.use_cap = False

        # probe and target different colors
        self.match_probe_and_target_color = False

        # Block is the linker object
        # self.num_blocks = 1
        # self.middle_scale_gradient = 0.0

    def clear_static_data(self) -> None:
        Dominoes.clear_static_data(self)

    def _write_static_data(self, static_group: h5py.Group) -> None:
        Dominoes._write_static_data(self, static_group)

    @staticmethod
    def get_controller_label_funcs(classname = "Linking"):
        funcs = Dominoes.get_controller_label_funcs(classname)

        return funcs

    def _write_frame_labels(self,
                            frame_grp: h5py.Group,
                            resp: List[bytes],
                            frame_num: int,
                            sleeping: bool) -> Tuple[h5py.Group, List[bytes], int, bool]:

        labels, resp, frame_num, done = Dominoes._write_frame_labels(
            self, frame_grp, resp, frame_num, sleeping)

        return labels, resp, frame_num, done

    def _get_zone_location(self, scale):
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:

        cmds = Dominoes.get_per_frame_commands(self, resp=resp, frame=frame)
        
        return cmds

    def _build_intermediate_structure(self) -> List[dict]:
        if self.randomize_colors_across_trials:
            self.middle_color = self.random_color(exclude=self.target_color) if self.monochrome else None

        commands = []

        # Build a stand for the linker object
        commands.extend(self._build_stand())

        # Add the attacment object (i.e. what the links will be partly attached to)
        commands.extend(self._place_attachment())

        # Add the links
        commands.extend(self._add_links())
        
        # # set camera params
        # camera_y_aim = 0.5 * self.tower_height
        # self.camera_aim = arr_to_xyz([0.,camera_y_aim,0.])

        return commands

    def _build_stand(self) -> List[dict]:
        commands = []
        return commands

    def _place_attachment(self) -> List[dict]:
        commands = []
        return commands

    def _add_links(self) -> List[dict]:
        commands = []
        return commands

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > 300

if __name__ == "__main__":

    args = get_linking_args("linking")

    LC = Linking(
        link_objects=args.middle,
        link_scale_range=args.mscale,
        num_links=args.num_middle_objects,
        
        # domino specific
        target_zone=args.zone,
        zone_location=args.zlocation,
        zone_scale_range=args.zscale,
        zone_color=args.zcolor,
        zone_friction=args.zfriction,        
        target_objects=args.target,
        probe_objects=args.probe,
        middle_objects=args.middle,
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
        LC.run(num=args.num,
               output_dir=args.dir,
               temp_path=args.temp,
               width=args.width,
               height=args.height,
               save_passes=args.save_passes.split(','),
               save_movies=args.save_movies,
               save_labels=args.save_labels,               
               args_dict=vars(args)
        )
    else:
        LC.communicate({"$type": "terminate"})
