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
from tdw_physics.target_controllers.dominoes import get_args as get_domino_args
from tdw_physics.target_controllers.towers import Tower, get_tower_args
from tdw_physics.postprocessing.labels import is_trial_valid

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}


def get_tower_args(dataset_dir: str, parse=True):
    """
    Combine Tower-specific args with general Dominoes args
    """
    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')

    parser.add_argument("--remove_target",
                        type=int_or_bool,
                        default=1,
                        help="Whether to remove the target object")
    parser.add_argument("--ramp",
                        type=int,
                        default=0,
                        help="Whether to place the probe object on the top of a ramp")    
    parser.add_argument("--collision_axis_length",
                        type=float,
                        default=3.0,
                        help="How far to put the probe and target")
    parser.add_argument("--num_blocks",
                        type=int,
                        default=3,
                        help="Number of rectangular blocks to build the tower base with")
    parser.add_argument("--mscale",
                        type=str,
                        default="[0.5,0.5]",
                        help="Scale or scale range for rectangular blocks to sample from")
    parser.add_argument("--mgrad",
                        type=float,
                        default=0.0,
                        help="Size of block scale gradient going from top to bottom of tower")
    parser.add_argument("--tower_cap",
                        type=none_or_str,
                        default="bowl",
                        help="Object types to use as a capper on the tower")
    parser.add_argument("--spacing_jitter",
                        type=float,
                        default=0.25,
                        help="jitter in how to space middle objects, as a fraction of uniform spacing")
    parser.add_argument("--mrot",
                        type=str,
                        default="[-45,45]",
                        help="comma separated list of initial middle object rotation values")
    parser.add_argument("--mmass",
                        type=str,
                        default="2.0",
                        help="comma separated list of initial middle object rotation values")    
    parser.add_argument("--middle",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible middle objects")
    parser.add_argument("--probe",
                        type=str,
                        default="sphere",
                        help="comma-separated list of possible target objects")
    parser.add_argument("--pmass",
                        type=str,
                        default="3.0",
                        help="scale of probe objects")
    parser.add_argument("--pscale",
                        type=str,
                        default="0.3",
                        help="scale of probe objects")
    parser.add_argument("--tscale",
                        type=str,
                        default="[0.5,0.5]",
                        help="scale of target objects")
    parser.add_argument("--zone",
                        type=none_or_str,
                        default="cube",
                        help="type of zone object")        
    parser.add_argument("--zscale",
                        type=str,
                        default="3.0,0.01,3.0",
                        help="scale of target zone")    
    parser.add_argument("--fscale",
                        type=str,
                        default="1.0",
                        help="range of scales to apply to push force")
    parser.add_argument("--frot",
                        type=str,
                        default="[-0,0]",
                        help="range of angles in xz plane to apply push force")
    parser.add_argument("--foffset",
                        type=str,
                        default="0.0,0.5,0.0",
                        help="offset from probe centroid from which to apply force, relative to probe scale")
    parser.add_argument("--fjitter",
                        type=float,
                        default=0.0,
                        help="jitter around object centroid to apply force")
    parser.add_argument("--fwait",
                        type=none_or_str,
                        default="[15,15]",
                        help="How many frames to wait before applying the force")        
    parser.add_argument("--camera_distance",
                        type=float,
                        default=3.0,
                        help="radial distance from camera to centerpoint")
    parser.add_argument("--camera_min_angle",
                        type=float,
                        default=0,
                        help="minimum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_max_angle",
                        type=float,
                        default=90,
                        help="maximum angle of camera rotation around centerpoint")

    # for generating training data without zones, targets, caps, and at lower resolution
    parser.add_argument("--training_data_mode",
                        action="store_true",
                        help="Overwrite some parameters to generate training data without target objects, zones, etc.")

    def postprocess(args):

        # whether to use a cap object on the tower
        if args.tower_cap is not None:
            cap_list = args.tower_cap.split(',')
            assert all([t in MODEL_NAMES for t in cap_list]), \
                "All target object names must be elements of %s" % MODEL_NAMES
            args.tower_cap = cap_list
        else:
            args.tower_cap = []


        return args

    args = parser.parse_args()
    args = domino_postproc(args)
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
        args.tower_cap = MODEL_NAMES

    return args

class Linking(Tower):
    
    def __init__(self,
                 port: int = 1071,
                 num_blocks=3,
                 middle_scale_range=[0.5,0.5],
                 middle_scale_gradient=0.0,
                 tower_cap=[],
                 use_ramp=True,
                 **kwargs):

        super().__init__(port=port, middle_scale_range=middle_scale_range, **kwargs)

        self.use_ramp = use_ramp
        self.use_cap False

        # probe and target different colors
        self.match_probe_and_target_color = False

        # Block is the linker object
        # self.num_blocks = 1
        # self.middle_scale_gradient = 0.0

    def clear_static_data(self) -> None:
        MultiDominoes.clear_static_data(self)

    def _write_static_data(self, static_group: h5py.Group) -> None:
        MultiDominoes._write_static_data(self, static_group)

    @staticmethod
    def get_controller_label_funcs(classname = "Linking"):
        funcs = Dominoes.get_controller_label_funcs(classname)

        return funcs

    def _write_frame_labels(self,
                            frame_grp: h5py.Group,
                            resp: List[bytes],
                            frame_num: int,
                            sleeping: bool) -> Tuple[h5py.Group, List[bytes], int, bool]:

        labels, resp, frame_num, done = MultiDominoes._write_frame_labels(
            self, frame_grp, resp, frame_num, sleeping)

        return labels, resp, frame_num, done

    def _get_zone_location(self, scale):
        return {"x": 0.0, "y": 0.0, "z": 0.0}


    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:

        cmds = super().get_per_frame_commands(resp, frame)
        
        return cmds

    def _build_intermediate_structure(self) -> List[dict]:
        if self.randomize_colors_across_trials:
            self.middle_color = self.random_color(exclude=self.target_color) if self.monochrome else None

        commands = []

        # Build a stand for the linker object
        commands.extend(self._build_stand())

        # Add the linker object (i.e. what the links will be partly attached to)
        commands.extend(self._place_linker())

        # Add the links
        commands.extend(self._add_links())
        
        # # set camera params
        # camera_y_aim = 0.5 * self.tower_height
        # self.camera_aim = arr_to_xyz([0.,camera_y_aim,0.])

        return commands

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > 300
        # return (frame > 750) or (self.fall_frame is not None and ((frame - 60) > self.fall_frame))

if __name__ == "__main__":

    args = get_tower_args("towers")

    TC = Tower(
        # tower specific
        num_blocks=args.num_blocks,
        tower_cap=args.tower_cap,
        spacing_jitter=args.spacing_jitter,
        middle_rotation_range=args.mrot,
        middle_scale_range=args.mscale,
        middle_mass_range=args.mmass,
        middle_scale_gradient=args.mgrad,
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
        TC.run(num=args.num,
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
        TC.communicate({"$type": "terminate"})
