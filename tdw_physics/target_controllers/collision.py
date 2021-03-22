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
from tdw_physics.util import MODEL_LIBRARIES, get_parser, xyz_to_arr, arr_to_xyz, str_to_xyz

from tdw_physics.target_controllers.dominoes import Dominoes, MultiDominoes, get_args, none_or_str, none_or_int
from tdw_physics.postprocessing.labels import is_trial_valid

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}

def get_collision_args(dataset_dir: str, parse=True):

    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')

    ## Changed defaults
    ### zone
    parser.add_argument("--zscale",
                        type=str,
                        default="3.0,0.01,3.0",
                        help="scale of target zone")

    parser.add_argument("--zone",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible target zone shapes")

    ### probe
    parser.add_argument("--probe",
                        type=str,
                        default="sphere",
                        help="comma-separated list of possible probe objects")

    parser.add_argument("--pscale",
                        type=str,
                        default="0.5,0.5,0.5",
                        help="scale of probe objects")

    ### force
    parser.add_argument("--fscale",
                        type=str,
                        default="[2.0,8.0]",
                        help="range of scales to apply to push force")

    parser.add_argument("--frot",
                        type=str,
                        default="[-20,20]",
                        help="range of angles in xz plane to apply push force")

    parser.add_argument("--foffset",
                        type=str,
                        default="0.0,0.8,0.0",
                        help="offset from probe centroid from which to apply force, relative to probe scale")

    parser.add_argument("--fjitter",
                        type=float,
                        default=0.5,
                        help="jitter around object centroid to apply force")
    ###target
    parser.add_argument("--target",
                        type=str,
                        default="pipe,cube,pentagon",
                        help="comma-separated list of possible target objects")

    parser.add_argument("--tscale",
                        type=str,
                        default="0.25,0.5,0.25",
                        help="scale of target objects")

    ### layout
    parser.add_argument("--collision_axis_length",
                        type=float,
                        default=1.0,
                        help="Length of spacing between probe and target objects at initialization.")


    def postprocess(args):
        return args

    args = parser.parse_args()
    args = domino_postproc(args)
    args = postprocess(args)

    return args

class Collision(MultiDominoes):

    def __init__(self,
                 port: int = 1071,
                #  bridge_height=1.0,
                 **kwargs):
        # initialize everything in common w / Multidominoes
        super().__init__(port=port, **kwargs)

        # do some Barrier-specific stuff
        # self.bridge_height = bridge_height

    def get_trial_initialization_commands(self) -> List[dict]:
        """This is where we string together the important commands of the controller in order"""
        # return super().get_trial_initialization_commands()
        commands = []

        # randomization across trials
        if not(self.randomize):
            self.trial_seed = (self.MAX_TRIALS * self.seed) + self._trial_num
            random.seed(self.trial_seed)
        else:
            self.trial_seed = -1 # not used

        # Choose and place the target zone.
        commands.extend(self._place_target_zone())

        # Choose and place a target object.
        commands.extend(self._place_target_object())

        # Set the probe color
        if self.probe_color is None:
            self.probe_color = self.target_color if (self.monochrome and self.match_probe_and_target_color) else None

        # Choose, place, and push a probe object.
        commands.extend(self._place_and_push_probe_object())

        # Build the intermediate structure that captures some aspect of "intuitive physics."
        commands.extend(self._build_intermediate_structure())

        # Teleport the avatar to a reasonable position 
        a_pos = self.get_random_avatar_position(radius_min=self.camera_radius,
                                                radius_max=self.camera_radius,
                                                angle_min=self.camera_min_angle,
                                                angle_max=self.camera_max_angle,
                                                y_min=self.camera_min_height,
                                                y_max=self.camera_max_height,
                                                center=TDWUtils.VECTOR3_ZERO)

        commands.extend([
            {"$type": "teleport_avatar_to",
             "position": a_pos},
            {"$type": "look_at_position",
             "position": self.camera_aim},
            {"$type": "set_focus_distance",
             "focus_distance": TDWUtils.get_distance(a_pos, self.camera_aim)}
        ])

        self.camera_position = a_pos
        self.camera_rotation = np.degrees(np.arctan2(a_pos['z'], a_pos['x']))
        dist = TDWUtils.get_distance(a_pos, self.camera_aim)
        self.camera_altitude = np.degrees(np.arcsin((a_pos['y'] - self.camera_aim['y'])/dist))

        # Place distractor objects in the background
        commands.extend(self._place_background_distractors())

        # Place occluder objects in the background
        commands.extend(self._place_occluders())

        return commands

    def _build_intermediate_structure(self) -> List[dict]:

        # print("middle color", self.middle_color)
        if self.randomize_colors_across_trials:
            self.middle_color = self.random_color(exclude=self.target_color) if self.monochrome else None

        commands = []

        # Go nuts
        # commands.extend(self._place_barrier_foundation())
        # commands.extend(self._build_bridge())

        return commands

    # def _place_barrier_foundation(self):
    #     return []

    # def _build_bridge(self):
    #     return []

    def _get_zone_location(self, scale):
        """Where to place the target zone?"""
        return {
            "x": 1 * self.collision_axis_length,# + scale["x"] + 0.1,
            "y": 0.0 if not self.remove_zone else 10.0,
            "z": 0.0 if not self.remove_zone else 10.0
        }

    def clear_static_data(self) -> None:
        Dominoes.clear_static_data(self)

        # clear some other stuff

    def _write_static_data(self, static_group: h5py.Group) -> None:
        Dominoes._write_static_data(self, static_group)

        # static_group.create_dataset("bridge_height", data=self.bridge_height)


    @staticmethod
    def get_controller_label_funcs(classname = "Collision"):

        funcs = Dominoes.get_controller_label_funcs(classname)

        return funcs
    

if __name__ == "__main__":
    import platform, os
    
    args = get_collision_args("collision")
    
    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":0." + str(args.gpu)
        else:
            os.environ["DISPLAY"] = ":0"

    ColC = Collision(
        # middle_objects=args.middle,
        # bridge_height=args.bridge_height
        room=args.room,
        num_middle_objects=args.num_middle_objects,
        randomize=args.random,
        seed=args.seed,
        target_zone=args.zone,
        zone_location=args.zlocation,
        zone_scale_range=args.zscale,
        zone_color=args.zcolor,
        zone_material=args.zmaterial,
        zone_friction=args.zfriction,
        target_objects=args.target,
        probe_objects=args.probe,
        middle_objects=args.middle,
        target_scale_range=args.tscale,
        target_rotation_range=args.trot,
        probe_rotation_range=args.prot,
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
        spacing_jitter=args.spacing_jitter,
        lateral_jitter=args.lateral_jitter,
        middle_scale_range=args.mscale,
        middle_rotation_range=args.mrot,
        middle_mass_range=args.mmass,
        horizontal=args.horizontal,
        remove_target=bool(args.remove_target),
        remove_zone=bool(args.remove_zone),
        ## not scenario-specific
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
        distractor_types=args.distractor,
        distractor_categories=args.distractor_categories,
        num_distractors=args.num_distractors,
        occluder_types=args.occluder,
        occluder_categories=args.occluder_categories,
        num_occluders=args.num_occluders,
        occlusion_scale=args.occlusion_scale,
        remove_middle=args.remove_middle,
        # ramp_color=args.rcolor
    )

    if bool(args.run):
        ColC.run(num=args.num,
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
        ColC.communicate({"$type": "terminate"})
