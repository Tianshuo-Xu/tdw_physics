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

'''
The linking controller generats stims in which the target object is 
    linked to, or partially contained within a base object. A probe object is launched at the base
    and *may* displace the target from the base
This controller pulls heavily from dominoes and tower controllers
The logic is much the same, but with slightly different params
Key (different) params;
    "attachment" is the linking object (what is being linked to)
    "middle" type of objects bing linked
'''

'''
Arguments:
  Linked objects
    --middle: 'torus'
    --mscale: "0.5,0.5,0.5", "0.7,0.7,0.7"
    --num_middle_range: "[1,4]"
    --spacing_jitter: 0.0,0.1
  linking Container
    --attachment: cylinder, "triangular_prism"
    --ascale: "0.5,0.5,0.5", "0.7,0.7,0.7", "0.9,0.9,0.9"
  Base Container
    --base: None,"cylinder"
    --bscale: "0.5", "0.9"
  Probe
    --fscale: "5.0", "7.0", "9.0"
'''

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
                        default="0.5,0.5,0.5",
                        help="The xyz scale ranges for each link object")
    parser.add_argument("--mmass",
                        type=none_or_str,
                        default="2.0",
                        help="The mass range of each link object")    
    parser.add_argument("--num_middle_range",
                        type=str,
                        default="[1,6]",
                        help="How many links to use")
    parser.add_argument("--spacing_jitter",
                        type=float,
                        default=0.0,
                        help="jitter in how to space middle objects, as a fraction of uniform spacing")    

    parser.add_argument("--target_link_range",
                        type=none_or_str,
                        default=None,
                        help="Which link to use as the target object. None is random, -1 is no target")

    parser.add_argument("--attachment",
                        type=none_or_str,
                        default="cylinder",
                        help="Which type of object to use as the attachment")
    parser.add_argument("--ascale",
                        type=none_or_str,
                        default="0.2,2.0,0.2",
                        help="Scale range (xyz) for attachment object")
    parser.add_argument("--amass",
                        type=none_or_str,
                        default="[3.0,3.0]",
                        help="Mass range for attachment object")
    parser.add_argument("--acolor",
                        type=none_or_str,
                        default="0.8,0.8,0.8",
                        help="Color for attachment object")
    parser.add_argument("--amaterial",
                        type=none_or_str,
                        default="wood_european_ash",
                        help="Material for attachment object")
    parser.add_argument("--attachment_fixed",
                        action="store_true",
                        help="Whether the attachment object will be fixed to the base or floor")
    parser.add_argument("--attachment_capped",
                        action="store_true",
                        help="Whether the attachment object will have a fixed cap like the base")    

    # base
    parser.add_argument("--base",
                        type=none_or_str,
                        default="cylinder",
                        help="Which type of object to use as the base")
    parser.add_argument("--bscale",
                        type=none_or_str,
                        default="0.5",
                        help="Scale range (xyz) for base object")
    parser.add_argument("--bmass",
                        type=none_or_str,
                        default="[3.0,3.0]",
                        help="Mass range for base object")
    parser.add_argument("--bcolor",
                        type=none_or_str,
                        default="0.8,0.8,0.8",
                        help="Color for base object")
    parser.add_argument("--bmaterial",
                        type=none_or_str,
                        default="wood_european_ash",
                        help="Material for base object")

    # ramp
    parser.add_argument("--ramp",
                        type=int_or_bool,
                        default=0,
                        help="Whether to place the probe object on the top of a ramp")
    parser.add_argument("--rheight",
                        type=none_or_str,
                        default=0.5,
                        help="Height of the ramp base")        

    # dominoes
    parser.add_argument("--collision_axis_length",
                        type=float,
                        default=2.0,
                        help="How far to put the probe and target")    

    # camera
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
    parser.add_argument("--camera_min_height",
                        type=float,
                        default=1.5,
                         help="min height of camera")
    parser.add_argument("--camera_max_height",
                        type=float,
                        default=2.5,
                        help="max height of camera")
    

    # for generating training data without zones, targets, caps, and at lower resolution
    parser.add_argument("--training_data_mode",
                        action="store_true",
                        help="Overwrite some parameters to generate training data without target objects, zones, etc.")

    def postprocess(args):

        # parent postprocess
        args = tower_postproc(args)

        # ramp
        args.rheight = handle_random_transform_args(args.rheight)        

        # num links
        args.num_middle_range = handle_random_transform_args(args.num_middle_range)
        
        # target
        args.target_link_range = handle_random_transform_args(args.target_link_range)
        
        # attachment
        args.ascale = handle_random_transform_args(args.ascale)
        args.amass = handle_random_transform_args(args.amass)
        if args.acolor is not None:
            args.acolor = [float(c) for c in args.acolor.split(',')]
        if args.attachment is not None:
            args.attachment = args.attachment.split(',')

        # base
        args.bscale = handle_random_transform_args(args.bscale)
        args.bmass = handle_random_transform_args(args.bmass)
        if args.bcolor is not None:
            args.bcolor = [float(c) for c in args.bcolor.split(',')]
        if args.base is not None:
            args.base = args.base.split(',')

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

    STANDARD_BLOCK_SCALE = {"x": 0.5, "y": 0.5, "z": 0.5}
    STANDARD_MASS_FACTOR = 0.25 
    
    def __init__(self,
                 port: int = None,
                 
                 # stand base
                 use_base=False,
                 base_object='cube',
                 base_scale_range=0.5,
                 base_mass_range=3.0,
                 base_color=[0.8,0.8,0.8],
                 base_material=None,
                 
                 # what object the links attach to
                 use_attachment=True,
                 attachment_object='cylinder',
                 attachment_scale_range={'x': 0.2, 'y': 2.0, 'z': 0.2},
                 attachment_mass_range=3.0,
                 attachment_fixed_to_base=False,
                 attachment_color=[0.8,0.8,0.8],
                 attachment_material=None,
                 use_cap=False,
                 
                 # what the links are, how many, and which is the target
                 link_objects='torus',
                 link_scale_range=0.5,
                 link_scale_gradient=0.0,
                 link_rotation_range=[0,0],
                 link_mass_range=2.0,
                 num_link_range=[1,6],
                 target_link_range=None,

                 # generic
                 use_ramp=True,
                 **kwargs):

        super().__init__(port=port, tower_cap=[], **kwargs)

        self.use_ramp = use_ramp

        # probe and target different colors
        self.match_probe_and_target_color = False

        # base
        self.use_base = use_base
        self._base_types = self.get_types(base_object) if self.use_base else self._target_types
        self.base_scale_range = base_scale_range
        self.base_mass_range = base_mass_range
        self.base_color = base_color
        self.base_material = base_material

        # attachment
        self.use_attachment = use_attachment        
        self._attachment_types = self.get_types(attachment_object) if self.use_attachment else self._target_types
        self.attachment_scale_range = attachment_scale_range
        self.attachment_color = attachment_color or self.middle_color
        self.attachment_mass_range = attachment_mass_range
        self.attachment_material = attachment_material
        self.use_cap = use_cap        

        # whether it'll be fixed to the base
        self.attachment_fixed_to_base = attachment_fixed_to_base        

        # links are the middle objects
        self.set_middle_types(link_objects)        
        self.num_link_range = num_link_range
        self.middle_scale_range = link_scale_range        
        self.middle_mass_range = link_mass_range
        self.middle_rotation_range = link_rotation_range
        self.middle_scale_gradient = link_scale_gradient
        self.target_link_range = target_link_range

    def clear_static_data(self) -> None:
        Dominoes.clear_static_data(self)

        self.tower_height = 0.0
        self.target_link_idx = None
        self.base = None
        self.attachment = None

    def _write_static_data(self, static_group: h5py.Group) -> None:
        Dominoes._write_static_data(self, static_group)

        static_group.create_dataset("base_id", data=self.base_id)
        static_group.create_dataset("use_base", data=self.use_base)        
        static_group.create_dataset("base_type", data=self.base_type)
        static_group.create_dataset("attachment_id", data=self.attachment_id)
        static_group.create_dataset("attachent_type", data=self.attachment_type)
        static_group.create_dataset("use_attachment", data=self.use_attachment)                
        static_group.create_dataset("link_type", data=self.middle_type)
        static_group.create_dataset("num_links", data=self.num_links)
        static_group.create_dataset("num_middle_objects", data=self.num_links)                
        static_group.create_dataset("target_link_idx", data=self.target_link_idx)
        static_group.create_dataset("attachment_fixed", data=self.attachment_fixed_to_base)
        static_group.create_dataset("use_cap", data=self.use_cap)                        

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
        commands.extend(self._build_base(height=self.tower_height, as_cap=False))

        # Add the attacment object (i.e. what the links will be partly attached to)
        commands.extend(self._place_attachment())

        # Add the links
        commands.extend(self._add_links())
        
        # # set camera params
        camera_y_aim = 0.5 * self.tower_height
        self.camera_aim = arr_to_xyz([0.,camera_y_aim,0.])

        return commands

    def _build_base(self, height, as_cap=False) -> List[dict]:
        
        commands = []

        record, data = self.random_primitive(
            self._base_types,
            scale=self.base_scale_range,
            color=self.base_color,
            exclude_color=self.target_color,
            add_data=self.use_base)
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        if as_cap:
            self.cap_id = data['id']
            self.cap_type = data['name']
        else:
            self.base_id = data['id']
            self.base_type = data['name']

        mass = random.uniform(*get_range(self.base_mass_range))
        mass *= (np.prod(xyz_to_arr(scale)) / np.prod(xyz_to_arr(self.STANDARD_BLOCK_SCALE)))

        print("base mass", mass)        

        commands.extend(
            self.add_physics_object(
                record=record,
                position={
                    "x": 0.,
                    "y": height,
                    "z": 0.
                },
                rotation={"x":0.,"y":0.,"z":0.},
                mass=mass,
                dynamic_friction=0.5,
                static_friction=0.5,
                bounciness=0,
                o_id=o_id,
                add_data=self.use_base
            ))

        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.base_material)))

        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": o_id},
            {"$type": "scale_object",
             "scale_factor": scale,
             "id": o_id}])

        if self.use_base:
            if self.base_type not in ['pipe', 'torus']:
                b_len, b_height, b_dep = self.get_record_dimensions(record)
                self.tower_height += b_height * scale['y']
        else:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]
            self.tower_height = 0.0

        return commands

    def _place_attachment(self) -> List[dict]:
        commands = []

        record, data = self.random_primitive(
            self._attachment_types,
            scale=self.attachment_scale_range,
            color=self.attachment_color,
            exclude_color=self.target_color,
            add_data=self.use_attachment)

        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        self.attachment = record
        self.attachment_id = data['id']
        self.attachment_type = data['name']

        mass = random.uniform(*get_range(self.attachment_mass_range))
        mass *= (np.prod(xyz_to_arr(scale)) / np.prod(xyz_to_arr(self.STANDARD_BLOCK_SCALE)))
        if self.attachment_type == 'cylinder':
            mass *= (np.pi / 4.0)
        elif self.attachment_type == 'cone':
            mass *= (np.pi / 12.0)

        print("attachment mass", mass)

        commands.extend(
            self.add_physics_object(
                record=record,
                position={
                    "x": 0.,
                    "y": self.tower_height,
                    "z": 0.
                },
                rotation={"x":0.,"y":0.,"z":0.},
                mass=mass,
                dynamic_friction=0.5,
                static_friction=0.5,
                bounciness=0,
                o_id=o_id,
                add_data=self.use_attachment
            ))

        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.attachment_material)))

        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": o_id},
            {"$type": "scale_object",
             "scale_factor": scale,
             "id": o_id}])

        # for an attachment that is wider at base, better to place links a little higher
        a_len, a_height, a_dep = self.get_record_dimensions(record)        
        if self.attachment_type == 'cone' and self.use_attachment:
            self.tower_height += 0.25 * a_height * (np.sqrt(scale['x']**2 + scale['z']**2) / scale['y'])

        if not self.use_attachment:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]
            

        # fix it to ground or block
        if self.attachment_fixed_to_base and self.use_attachment:
            # make it kinematic
            if self.use_base:
                commands.append({
                    "$type": "add_fixed_joint",
                    "parent_id": self.base_id,
                    "id": o_id})
            elif not self.use_base: # make kinematic
                commands.extend([
                    {"$type": "set_object_collision_detection_mode",
                     "mode": "continuous_speculative",
                     "id": o_id},
                    {"$type": "set_kinematic_state",
                     "id": o_id,
                     "is_kinematic": True,
                     "use_gravity": True}])

        # add a cap
        if self.use_cap and self.use_attachment:
            commands.extend(self._build_base(
                height=(self.tower_height + a_height * scale['y']),
                as_cap=True))
            if self.attachment_fixed_to_base:
                commands.append({
                    "$type": "add_fixed_joint",
                    "parent_id": o_id,
                    "id": self.cap_id})                
            

        return commands        

    def _add_links(self) -> List[dict]:
        commands = []

        # select how many links
        self.num_links = self.num_blocks = random.choice(range(*self.num_link_range))

        # build a "tower" out of the links
        commands.extend(self._build_stack())

        # change one of the links to the target object
        commands.extend(self._switch_target_link())

        return commands

    def _switch_target_link(self) -> List[dict]:

        commands = []

        if self.target_link_range is None:
            self.target_link_idx = random.choice(range(self.num_links))
        elif hasattr(self.target_link_range, '__len__'):
            self.target_link_idx = int(random.choice(range(*get_range(self.target_link_range))))
            self.target_link_idx = min(self.target_link_idx, self.num_links - 1)
        elif isinstance(self.target_link_range, (int, float)):
            self.target_link_idx = int(self.target_link_range)
        else:
            return []

        print("target is link idx %d" % self.target_link_idx)
            
        if int(self.target_link_idx) not in range(self.num_links):
            print("no target link")
            return [] # no link is the target

        record, data = self.blocks[self.target_link_idx]
        o_id = data['id']

        # update the data so that it's the target
        if self.target_color is None:
            self.target_color = self.random_color()
        data['color'] = self.target_color
        self._replace_target_with_object(record, data)

        # add the commands to change the material and color
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.target_material)))

        # Scale the object and set its color.
        rgb = data['color']
        commands.extend([
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": o_id}])

        return commands


    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > 450

if __name__ == "__main__":

    args = get_linking_args("linking")

    LC = Linking(
        # links
        link_objects=args.middle,
        link_scale_range=args.mscale,
        link_scale_gradient=args.mgrad,
        link_rotation_range=args.mrot,
        link_mass_range=args.mmass,
        num_link_range=args.num_middle_range,
        target_link_range=args.target_link_range,
        spacing_jitter=args.spacing_jitter,

        # base
        use_base=(args.base is not None),
        base_object=args.base,
        base_scale_range=args.bscale,
        base_mass_range=args.bmass,
        base_color=args.bcolor,
        base_material=args.bmaterial,
        
        # attachment
        use_attachment=(args.attachment is not None),
        attachment_object=args.attachment,
        attachment_scale_range=args.ascale,
        attachment_mass_range=args.amass,
        attachment_color=args.acolor,
        attachment_material=args.amaterial,
        attachment_fixed_to_base=args.attachment_fixed,
        use_cap=args.attachment_capped,

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
        ramp_color=args.rcolor,
        ramp_base_height_range=args.rheight,
        flex_only=args.only_use_flex_objects,
        no_moving_distractors=args.no_moving_distractors       
    )

    if bool(args.run):
        LC.run(num=args.num,
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
        LC.communicate({"$type": "terminate"})
