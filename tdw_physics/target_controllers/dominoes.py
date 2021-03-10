from argparse import ArgumentParser
import h5py
import json
import copy
import importlib
import numpy as np
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
from tdw_physics.util import MODEL_LIBRARIES, get_parser, xyz_to_arr, arr_to_xyz, str_to_xyz
from tdw_physics.postprocessing.labels import get_all_label_funcs


MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}
ALL_NAMES = [r.name for r in MODEL_LIBRARIES['models_full.json'].records]

def none_or_str(value):
    if value == 'None':
        return None
    else:
        return value

def get_args(dataset_dir: str, parse=True):
    """
    Combine Domino-specific arguments with controller-common arguments
    """
    common = get_parser(dataset_dir, get_help=False)
    parser = ArgumentParser(parents=[common], add_help=parse, fromfile_prefix_chars='@')

    parser.add_argument("--num_middle_objects",
                        type=int,
                        default=3,
                        help="The number of middle objects to place")
    parser.add_argument("--zone",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible target zone shapes")
    parser.add_argument("--target",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible target objects")
    parser.add_argument("--probe",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible target objects")
    parser.add_argument("--middle",
                        type=str,
                        default=None,
                        help="comma-separated list of possible middle objects; default to same as target")
    parser.add_argument("--ramp",
                        type=int,
                        default=0,
                        help="Whether to place the probe object on the top of a ramp")
    parser.add_argument("--zscale",
                        type=str,
                        default="0.5,0.01,2.0",
                        help="scale of target zone")
    parser.add_argument("--zlocation",
                        type=none_or_str,
                        default=None,
                        help="Where to place the target zone. None will default to a scenario-specific place.")
    parser.add_argument("--zfriction",
                        type=float,
                        default=0.1,
                        help="Static and dynamic friction on the target zone.")    
    parser.add_argument("--tscale",
                        type=str,
                        default="0.1,0.5,0.25",
                        help="scale of target objects")
    parser.add_argument("--trot",
                        type=str,
                        default="[0,0]",
                        help="comma separated list of initial target rotation values")
    parser.add_argument("--mrot",
                        type=str,
                        default="[-30,30]",
                        help="comma separated list of initial middle object rotation values")
    parser.add_argument("--prot",
                        type=str,
                        default="[0,0]",
                        help="comma separated list of initial probe rotation values")    
    parser.add_argument("--mscale",
                        type=str,
                        default=None,
                        help="Scale or scale range for middle objects")
    parser.add_argument("--mmass",
                        type=str,
                        default="2.0",
                        help="Scale or scale range for middle objects")
    parser.add_argument("--horizontal",
                        type=int,
                        default=0,
                        help="Whether to rotate middle objects horizontally")
    parser.add_argument("--pscale",
                        type=str,
                        default="0.1,0.5,0.25",
                        help="scale of probe objects")
    parser.add_argument("--pmass",
                        type=str,
                        default="2.0",
                        help="scale of probe objects")
    parser.add_argument("--fscale",
                        type=str,
                        default="2.0",
                        help="range of scales to apply to push force")
    parser.add_argument("--frot",
                        type=str,
                        default="[0,0]",
                        help="range of angles in xz plane to apply push force")
    parser.add_argument("--foffset",
                        type=str,
                        default="0.0,0.8,0.0",
                        help="offset from probe centroid from which to apply force, relative to probe scale")
    parser.add_argument("--fjitter",
                        type=float,
                        default=0.0,
                        help="jitter around object centroid to apply force")
    parser.add_argument("--fwait",
                        type=none_or_str,
                        default="[0,0]",
                        help="How many frames to wait before applying the force")    
    parser.add_argument("--color",
                        type=none_or_str,
                        default="1.0,0.0,0.0",
                        help="comma-separated R,G,B values for the target object color. None to random.")
    parser.add_argument("--zcolor",
                        type=none_or_str,
                        default="1.0,1.0,0.0",
                        help="comma-separated R,G,B values for the target zone color. None is random")
    parser.add_argument("--pcolor",
                        type=none_or_str,
                        default="0.0,1.0,1.0",
                        help="comma-separated R,G,B values for the probe object color. None is random.")
    parser.add_argument("--mcolor",
                        type=none_or_str,
                        default=None,
                        help="comma-separated R,G,B values for the middle object color. None is random.")
    parser.add_argument("--collision_axis_length",
                        type=float,
                        default=2.0,
                        help="Length of spacing between probe and target objects at initialization.")
    parser.add_argument("--spacing_jitter",
                        type=float,
                        default=0.2,
                        help="jitter in how to space middle objects, as a fraction of uniform spacing")
    parser.add_argument("--lateral_jitter",
                        type=float,
                        default=0.2,
                        help="lateral jitter in how to space middle objects, as a fraction of object width")
    parser.add_argument("--remove_target",
                        type=int,
                        default=0,
                        help="Don't actually put the target object in the scene.")
    parser.add_argument("--camera_distance",
                        type=float,
                        default=1.75,
                        help="radial distance from camera to centerpoint")
    parser.add_argument("--camera_min_height",
                        type=float,
                        default=0.75,
                         help="min height of camera")
    parser.add_argument("--camera_max_height",
                        type=float,
                        default=2.0,
                        help="max height of camera")
    parser.add_argument("--camera_min_angle",
                        type=float,
                        default=45,
                        help="minimum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_max_angle",
                        type=float,
                        default=225,
                        help="maximum angle of camera rotation around centerpoint")
    parser.add_argument("--material_types",
                        type=none_or_str,
                        default="Wood,Metal,Plastic",
                        help="Which class of materials to sample material names from")
    parser.add_argument("--tmaterial",
                        type=none_or_str,
                        default="parquet_wood_red_cedar",
                        help="Material name for target. If None, samples from material_type")
    parser.add_argument("--zmaterial",
                        type=none_or_str,
                        default="wood_european_ash",
                        help="Material name for target. If None, samples from material_type")
    parser.add_argument("--pmaterial",
                        type=none_or_str,
                        default="parquet_wood_red_cedar",
                        help="Material name for probe. If None, samples from material_type")
    parser.add_argument("--mmaterial",
                        type=none_or_str,
                        default="parquet_wood_red_cedar",
                        help="Material name for middle objects. If None, samples from material_type")
    parser.add_argument("--distractor",
                        type=none_or_str,
                        default="core",
                        help="The names or library of distractor objects to use")
    parser.add_argument("--distractor_categories",
                        type=none_or_str,
                        help="The categories of distractors to choose from (comma-separated)")
    parser.add_argument("--num_distractors",
                        type=int,
                        default=0,
                        help="The number of background distractor objects to place")
    parser.add_argument("--occluder",
                        type=none_or_str,
                        default="core",
                        help="The names or library of occluder objects to use")
    parser.add_argument("--occluder_categories",
                        type=none_or_str,
                        help="The categories of occluders to choose from (comma-separated)")
    parser.add_argument("--num_occluders",
                        type=int,
                        default=0,
                        help="The number of foreground occluder objects to place")
    parser.add_argument("--occlusion_scale",
                        type=float,
                        default=0.75,
                        help="The height of the occluders as a proportion of camera height")
    parser.add_argument("--remove_middle",
                        action="store_true",
                        help="Remove one of the middle dominoes scene.")    

    def postprocess(args):
        # choose a valid room
        assert args.room in ['box', 'tdw', 'house'], args.room

        # whether to set all objects same color
        args.monochrome = bool(args.monochrome)

        # scaling and rotating of objects
        args.zscale = handle_random_transform_args(args.zscale)
        args.zlocation = handle_random_transform_args(args.zlocation)
        args.tscale = handle_random_transform_args(args.tscale)
        args.trot = handle_random_transform_args(args.trot)
        args.pscale = handle_random_transform_args(args.pscale)
        args.pmass = handle_random_transform_args(args.pmass)
        args.prot = handle_random_transform_args(args.prot)        
        args.mscale = handle_random_transform_args(args.mscale)
        args.mrot = handle_random_transform_args(args.mrot)
        args.mmass = handle_random_transform_args(args.mmass)

        # the push force scale and direction
        args.fscale = handle_random_transform_args(args.fscale)
        args.frot = handle_random_transform_args(args.frot)
        args.foffset = handle_random_transform_args(args.foffset)
        args.fwait = handle_random_transform_args(args.fwait)

        args.horizontal = bool(args.horizontal)

        if args.zone is not None:
            zone_list = args.zone.split(',')
            assert all([t in MODEL_NAMES for t in zone_list]), \
                "All target object names must be elements of %s" % MODEL_NAMES
            args.zone = zone_list
        else:
            args.target = MODEL_NAMES

        if args.target is not None:
            targ_list = args.target.split(',')
            assert all([t in MODEL_NAMES for t in targ_list]), \
                "All target object names must be elements of %s" % MODEL_NAMES
            args.target = targ_list
        else:
            args.target = MODEL_NAMES

        if args.probe is not None:
            probe_list = args.probe.split(',')
            assert all([t in MODEL_NAMES for t in probe_list]), \
                "All target object names must be elements of %s" % MODEL_NAMES
            args.probe = probe_list
        else:
            args.probe = MODEL_NAMES

        if args.middle is not None:
            middle_list = args.middle.split(',')
            assert all([t in MODEL_NAMES for t in middle_list]), \
                "All target object names must be elements of %s" % MODEL_NAMES
            args.middle = middle_list

        if args.color is not None:
            rgb = [float(c) for c in args.color.split(',')]
            assert len(rgb) == 3, rgb
            args.color = rgb

        if args.zcolor is not None:
            rgb = [float(c) for c in args.zcolor.split(',')]
            assert len(rgb) == 3, rgb
            args.zcolor = rgb

        if args.pcolor is not None:
            rgb = [float(c) for c in args.pcolor.split(',')]
            assert len(rgb) == 3, rgb
            args.pcolor = rgb

        if args.mcolor is not None:
            rgb = [float(c) for c in args.mcolor.split(',')]
            assert len(rgb) == 3, rgb
            args.mcolor = rgb


        if args.material_types is None:
            args.material_types = MATERIAL_TYPES
        else:
            matlist = args.material_types.split(',')
            assert all ([m in MATERIAL_TYPES for m in matlist]), \
                "All material types must be elements of %s" % MATERIAL_TYPES
            args.material_types = matlist

        if args.distractor is None or args.distractor == 'full':
            args.distractor = ALL_NAMES
        elif args.distractor == 'core':
            args.distractor = [r.name for r in MODEL_LIBRARIES['models_core.json'].records]
        elif args.distractor in ['flex', 'primitives']:
            args.distractor = MODEL_NAMES
        else:
            d_names = args.distractor.split(',')
            args.distractor = [r for r in ALL_NAMES if any((nm in r for nm in d_names))]

        if args.occluder is None or args.occluder == 'full':
            args.occluder = ALL_NAMES
        elif args.occluder == 'core':
            args.occluder = [r.name for r in MODEL_LIBRARIES['models_core.json'].records]
        elif args.occluder in ['flex', 'primitives']:
            args.occluder = MODEL_NAMES
        else:
            o_names = args.occluder.split(',')
            args.occluder = [r for r in ALL_NAMES if any((nm in r for nm in o_names))]

        return args

    if not parse:
        return (parser, postprocess)
    else:
        args = parser.parse_args()
        args = postprocess(args)
        return args

class Dominoes(RigidbodiesDataset):
    """
    Drop a random Flex primitive object on another random Flex primitive object
    """

    MAX_TRIALS = 1000
    DEFAULT_RAMPS = [r for r in MODEL_LIBRARIES['models_full.json'].records if 'ramp_with_platform_30' in r.name]
    
    def __init__(self,
                 port: int = 1071,
                 room='box',
                 target_zone=['cube'],
                 zone_color=[0.0,0.5,1.0],
                 zone_location=None,
                 zone_scale_range=[0.5,0.001,0.5],
                 zone_friction=0.1,
                 probe_objects=MODEL_NAMES,
                 target_objects=MODEL_NAMES,
                 probe_scale_range=[0.2, 0.3],
                 probe_mass_range=[2.,7.],
                 probe_color=None,
                 probe_rotation_range=[0,0],
                 target_scale_range=[0.2, 0.3],
                 target_rotation_range=None,
                 target_color=None,
                 target_motion_thresh=0.01,
                 collision_axis_length=1.,
                 force_scale_range=[0.,8.],
                 force_angle_range=[-60,60],
                 force_offset={"x":0.,"y":0.5,"z":0.0},
                 force_offset_jitter=0.1,
                 force_wait=None,
                 remove_target=False,
                 camera_radius=1.0,
                 camera_min_angle=0,
                 camera_max_angle=360,
                 camera_min_height=1./3,
                 camera_max_height=2./3,
                 material_types=MATERIAL_TYPES,
                 target_material=None,
                 probe_material=None,
                 zone_material=None,
                 distractor_types=MODEL_NAMES,
                 distractor_categories=None,
                 num_distractors=0,
                 occluder_types=MODEL_NAMES,
                 occluder_categories=None,
                 num_occluders=0,
                 occlusion_scale=0.6,
                 use_ramp=False,
                 **kwargs):

        ## initializes static data and RNG
        super().__init__(port=port, **kwargs)

        ## which room to use
        self.room = room

        ## target zone
        self.set_zone_types(target_zone)
        self.zone_location = zone_location
        self.zone_color = zone_color
        self.zone_scale_range = zone_scale_range
        self.zone_material = zone_material
        self.zone_friction = zone_friction

        ## allowable object types
        self.set_probe_types(probe_objects)
        self.set_target_types(target_objects)
        self.material_types = material_types
        self.remove_target = remove_target

        # whether to use a ramp
        self.use_ramp = use_ramp

        ## object generation properties
        self.target_scale_range = target_scale_range
        self.target_color = target_color
        self.target_rotation_range = target_rotation_range
        self.target_material = target_material
        self.target_motion_thresh = target_motion_thresh

        self.probe_color = probe_color
        self.probe_scale_range = probe_scale_range
        self.probe_rotation_range = probe_rotation_range
        self.probe_mass_range = get_range(probe_mass_range)
        self.probe_material = probe_material
        self.match_probe_and_target_color = True

        self.middle_scale_range = target_scale_range

        ## Scenario config properties
        self.collision_axis_length = collision_axis_length
        self.force_scale_range = force_scale_range
        self.force_angle_range = force_angle_range
        self.force_offset = get_random_xyz_transform(force_offset)
        self.force_offset_jitter = force_offset_jitter
        self.force_wait_range = force_wait or [0,0]

        ## camera properties
        self.camera_radius = camera_radius
        self.camera_min_angle = camera_min_angle
        self.camera_max_angle = camera_max_angle
        self.camera_min_height = camera_min_height
        self.camera_max_height = camera_max_height
        self.camera_aim = {"x": 0., "y": 0.5, "z": 0.} # fixed aim

        ## distractors and occluders
        self.num_distractors = num_distractors
        self.distractor_types = self.get_types(
            distractor_types,
            libraries=["models_flex.json", "models_full.json", "models_special.json"],
            categories=distractor_categories)

        self.num_occluders = num_occluders
        self.occlusion_scale = occlusion_scale
        self.occluder_types = self.get_types(
            occluder_types,
            libraries=["models_flex.json", "models_full.json", "models_special.json"],
            categories=occluder_categories)


    def get_types(self, objlist, libraries=["models_flex.json"], categories=None):
        recs = []
        for lib in libraries:
            recs.extend(MODEL_LIBRARIES[lib].records)
        tlist = [r for r in recs if r.name in objlist]
        if categories is not None:
            if not isinstance(categories, list):
                categories = categories.split(',')
            tlist = [r for r in tlist if r.wcategory in categories]
        return tlist

    def set_probe_types(self, olist):
        tlist = self.get_types(olist)
        self._probe_types = tlist

    def set_target_types(self, olist):
        tlist = self.get_types(olist)
        self._target_types = tlist

    def set_zone_types(self, olist):
        tlist = self.get_types(olist)
        self._zone_types = tlist

    def get_material_name(self, material):

        if material is not None:
            if material in MATERIAL_TYPES:
                mat = random.choice(MATERIAL_NAMES[material])
            else:
                assert any((material in MATERIAL_NAMES[mtype] for mtype in self.material_types)), \
                    (material, self.material_types)
                mat = material
        else:
            mtype = random.choice(self.material_types)
            mat = random.choice(MATERIAL_NAMES[mtype])

        return mat

    def get_object_material_commands(self, record, object_id, material):
        commands = TDWUtils.set_visual_material(
            self, record.substructure, object_id, material, quality="high")
        return commands

    def clear_static_data(self) -> None:
        super().clear_static_data()

        ## scenario-specific metadata: object types and drop position
        self.target_type = None
        self.target_rotation = None
        self.target_position = None
        self.target_delta_position = None
        self.replace_target = False

        self.probe_type = None
        self.probe_mass = None
        self.push_force = None
        self.push_position = None
        self.force_wait = None

    @staticmethod
    def get_controller_label_funcs(classname = 'Dominoes'):

        funcs = super(Dominoes, Dominoes).get_controller_label_funcs(classname)
        funcs += get_all_label_funcs()

        def room(f):
            return str(np.array(f['static']['room'], dtype=str))
        def trial_seed(f):
            return int(np.array(f['static']['trial_seed']))
        def num_distractors(f):
            try:
                return int(len(f['static']['distractors']))
            except KeyError:
                return int(0)
        def num_occluders(f):
            try:
                return int(len(f['static']['occluders']))
            except KeyError:
                return int(0)
        def push_time(f):
            try:
                return int(np.array(f['static']['push_time']))
            except KeyError:
                return int(0)
        funcs += [room, trial_seed, push_time, num_distractors, num_occluders]
        
        return funcs

    def get_field_of_view(self) -> float:
        return 55

    def get_scene_initialization_commands(self) -> List[dict]:
        if self.room == 'box':
            add_scene = self.get_add_scene(scene_name="box_room_2018")
        elif self.room == 'tdw':
            add_scene = self.get_add_scene(scene_name="tdw_room")
        elif self.room == 'house':
            add_scene = self.get_add_scene(scene_name='archviz_house')
        return [add_scene,
                {"$type": "set_aperture",
                 "aperture": 8.0},
                {"$type": "set_post_exposure",
                 "post_exposure": 0.4},
                {"$type": "set_ambient_occlusion_intensity",
                 "intensity": 0.175},
                {"$type": "set_ambient_occlusion_thickness_modifier",
                 "thickness": 3.5}]

    def get_trial_initialization_commands(self) -> List[dict]:
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

        # Teleport the avatar to a reasonable position based on the drop height.
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

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:

        if (self.force_wait != 0) and frame == self.force_wait:
            print("applied %s at time step %d" % (self.push_cmd, frame))
            return [self.push_cmd]
        else:
            return []

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        # randomization
        static_group.create_dataset("room", data=self.room)
        static_group.create_dataset("seed", data=self.seed)
        static_group.create_dataset("randomize", data=self.randomize)
        static_group.create_dataset("trial_seed", data=self.trial_seed)
        static_group.create_dataset("trial_num", data=self._trial_num)        

        ## which objects are the zone, target, and probe
        static_group.create_dataset("zone_id", data=self.zone_id)
        static_group.create_dataset("target_id", data=self.target_id)
        static_group.create_dataset("probe_id", data=self.probe_id)

        ## color and scales of primitive objects
        static_group.create_dataset("target_type", data=self.target_type)
        static_group.create_dataset("target_rotation", data=xyz_to_arr(self.target_rotation))
        static_group.create_dataset("probe_type", data=self.probe_type)
        static_group.create_dataset("probe_mass", data=self.probe_mass)
        static_group.create_dataset("push_force", data=xyz_to_arr(self.push_force))
        static_group.create_dataset("push_position", data=xyz_to_arr(self.push_position))
        static_group.create_dataset("push_time", data=int(self.force_wait))

        # distractors and occluders
        static_group.create_dataset("distractors", data=[r.name for r in self.distractors.values()])
        static_group.create_dataset("occluders", data=[r.name for r in self.occluders.values()])

    def _write_frame(self,
                     frames_grp: h5py.Group,
                     resp: List[bytes],
                     frame_num: int) -> \
            Tuple[h5py.Group, h5py.Group, dict, bool]:
        frame, objs, tr, sleeping = super()._write_frame(frames_grp=frames_grp,
                                                         resp=resp,
                                                         frame_num=frame_num)
        # If this is a stable structure, disregard whether anything is actually moving.
        return frame, objs, tr, sleeping and not (frame_num < 150)

    def _update_target_position(self, resp: List[bytes], frame_num: int) -> None:
        if frame_num <= 0:
            self.target_delta_position = xyz_to_arr(TDWUtils.VECTOR3_ZERO)
        elif 'tran' in [OutputData.get_data_type_id(r) for r in resp[:-1]]:
            target_position_new = self.get_object_position(self.target_id, resp) or self.target_position
            try:
                self.target_delta_position += (target_position_new - xyz_to_arr(self.target_position))
                self.target_position = arr_to_xyz(target_position_new)
            except TypeError:
                print("Failed to get a new object position, %s" % target_position_new)

    def _write_frame_labels(self,
                            frame_grp: h5py.Group,
                            resp: List[bytes],
                            frame_num: int,
                            sleeping: bool) -> Tuple[h5py.Group, List[bytes], int, bool]:

        labels, resp, frame_num, done = super()._write_frame_labels(frame_grp, resp, frame_num, sleeping)

        # Whether this trial has a target or zone to track
        has_target = (not self.remove_target) or self.replace_target
        has_zone = not self.remove_zone
        labels.create_dataset("has_target", data=has_target)
        labels.create_dataset("has_zone", data=has_zone)
        if not (has_target or has_zone):
            return labels, done

        # Whether target moved from its initial position, and how much
        if has_target:
            self._update_target_position(resp, frame_num)
            has_moved = np.sqrt((self.target_delta_position**2).sum()) > self.target_motion_thresh
            labels.create_dataset("target_delta_position", data=self.target_delta_position)
            labels.create_dataset("target_has_moved", data=has_moved)

            # Whether target has fallen to the ground
            c_points, c_normals = self.get_object_environment_collision(
                self.target_id, resp)

            if frame_num <= 0:
                self.target_on_ground = False
                self.target_ground_contacts = c_points
            elif len(c_points) == 0:
                self.target_on_ground = False
            elif len(c_points) != len(self.target_ground_contacts):
                self.target_on_ground = True
            elif any([np.sqrt(((c_points[i] - self.target_ground_contacts[i])**2).sum()) > self.target_motion_thresh \
                      for i in range(min(len(c_points), len(self.target_ground_contacts)))]):
                self.target_on_ground = True

            labels.create_dataset("target_on_ground", data=self.target_on_ground)

        # Whether target has hit the zone
        if has_target and has_zone:
            c_points, c_normals = self.get_object_target_collision(
                self.target_id, self.zone_id, resp)
            target_zone_contact = bool(len(c_points))
            labels.create_dataset("target_contacting_zone", data=target_zone_contact)

        return labels, resp, frame_num, done

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > 300

    def get_rotation(self, rot_range):
        if rot_range is None:
            return {"x": 0,
                    "y": random.uniform(0, 360),
                    "z": 0.}
        else:
            return get_random_xyz_transform(rot_range)

    def get_y_rotation(self, rot_range):
        if rot_range is None:
            return self.get_rotation(rot_range)
        else:
            return {"x": 0.,
                    "y": random.uniform(*get_range(rot_range)),
                    "z": 0.}

    def get_push_force(self, scale_range, angle_range):
        # rotate a unit vector initially pointing in positive-x direction
        theta = np.radians(random.uniform(*get_range(angle_range)))
        push = np.array([np.cos(theta), 0., np.sin(theta)])

        # scale it
        push *= random.uniform(*get_range(scale_range))

        # convert to xyz
        return arr_to_xyz(push)

    def _get_zone_location(self, scale):
        return {
            "x": 0.5 * self.collision_axis_length + scale["x"] + 0.1,
            "y": 0.0 if not self.remove_zone else 10.0,
            "z": 0.0 if not self.remove_zone else 10.0
        }


    def _place_target_zone(self) -> List[dict]:

        # create a target zone (usually flat, with same texture as room)
        record, data = self.random_primitive(self._zone_types,
                                             scale=self.zone_scale_range,
                                             color=self.zone_color,
                                             add_data=True
        )
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        self.zone = record
        self.zone_type = data["name"]
        self.zone_color = rgb
        self.zone_id = o_id
        self.zone_scale = scale

        if any((s <= 0 for s in scale.values())):
            self.remove_zone = True
            self.scales = self.scales[:-1]
            self.colors = self.colors[:-1]
            self.model_names = self.model_names[:-1]
        else:
            self.remove_zone = False

        # place it just beyond the target object with an effectively immovable mass and high friction
        commands = []
        commands.extend(
            self.add_physics_object(
                record=record,
                position=(self.zone_location or self._get_zone_location(scale)),
                rotation=TDWUtils.VECTOR3_ZERO,
                mass=500,
                dynamic_friction=self.zone_friction,
                static_friction=(10.0 * self.zone_friction),
                bounciness=0,
                o_id=o_id,
                add_data=(not self.remove_zone)
            ))

        # set its material to be the same as the room
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.zone_material)))

        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": o_id},
            {"$type": "scale_object",
             "scale_factor": scale if not self.remove_zone else TDWUtils.VECTOR3_ZERO,
             "id": o_id}])

        # make it a "kinematic" object that won't move
        commands.extend([
            {"$type": "set_object_collision_detection_mode",
             "mode": "continuous_speculative",
             "id": o_id},
            {"$type": "set_kinematic_state",
             "id": o_id,
             "is_kinematic": True,
             "use_gravity": True}])            

        # get rid of it if not using a target object
        if self.remove_zone:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]

        return commands

    def _place_target_object(self) -> List[dict]:
        """
        Place a primitive object at one end of the collision axis.
        """

        # create a target object
        record, data = self.random_primitive(self._target_types,
                                             scale=self.target_scale_range,
                                             color=self.target_color,
                                             add_data=(not self.remove_target)
        )
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        self.target = record
        self.target_type = data["name"]
        self.target_color = rgb
        self.target_scale = self.middle_scale = scale
        self.target_id = o_id

        if any((s <= 0 for s in scale.values())):
            self.remove_target = True

        # Where to put the target
        if self.target_rotation is None:
            self.target_rotation = self.get_rotation(self.target_rotation_range)

        if self.target_position is None:
            self.target_position = {
                "x": 0.5 * self.collision_axis_length,
                "y": 0. if not self.remove_target else 10.0,
                "z": 0. if not self.remove_target else 10.0
            }

        # Commands for adding hte object
        commands = []
        commands.extend(
            self.add_physics_object(
                record=record,
                position=self.target_position,
                rotation=self.target_rotation,
                mass=2.0,
                dynamic_friction=0.5,
                static_friction=0.5,
                bounciness=0.0,
                o_id=o_id,
                add_data=(not self.remove_target)
            ))

        # Set the object material
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.target_material)))

        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": o_id},
            {"$type": "scale_object",
             "scale_factor": scale if not self.remove_target else TDWUtils.VECTOR3_ZERO,
             "id": o_id}])

        # If this scene won't have a target
        if self.remove_target:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]

        return commands

    def _place_and_push_probe_object(self) -> List[dict]:
        """
        Place a probe object at the other end of the collision axis, then apply a force to push it.
        """
        exclude = not (self.monochrome and self.match_probe_and_target_color)
        record, data = self.random_primitive(self._probe_types,
                                             scale=self.probe_scale_range,
                                             color=self.probe_color,
                                             exclude_color=(self.target_color if exclude else None),
                                             exclude_range=0.25)
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        self.probe = record
        self.probe_type = data["name"]
        self.probe_scale = scale
        self.probe_id = o_id

        # Add the object with random physics values
        commands = []

        ### TODO: better sampling of random physics values
        self.probe_mass = random.uniform(self.probe_mass_range[0], self.probe_mass_range[1])
        self.probe_initial_position = {"x": -0.5*self.collision_axis_length, "y": 0., "z": 0.}
        rot = self.get_y_rotation(self.probe_rotation_range)

        if self.use_ramp:
            commands.extend(self._place_ramp_under_probe())
        
        commands.extend(
            self.add_physics_object(
                record=record,
                position=self.probe_initial_position,
                rotation=rot,
                mass=self.probe_mass,
                # dynamic_friction=0.5,
                # static_friction=0.5,
                # bounciness=0.1,
                dynamic_friction=0.01,
                static_friction=0.01,
                bounciness=0,                
                o_id=o_id))

        # Set the probe material
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.probe_material)))


        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": o_id},
            {"$type": "scale_object",
             "scale_factor": scale,
             "id": o_id}])

        # Set its collision mode
        commands.extend([
            # {"$type": "set_object_collision_detection_mode",
            #  "mode": "continuous_speculative",
            #  "id": o_id},
            {"$type": "set_object_drag",
             "id": o_id,
             "drag": 0, "angular_drag": 0}])
            

        # Apply a force to the probe object
        self.push_force = self.get_push_force(
            scale_range=self.probe_mass * np.array(self.force_scale_range),
            angle_range=self.force_angle_range)
        self.push_force = self.rotate_vector_parallel_to_floor(
            self.push_force, -rot['y'], degrees=True)

        self.push_position = self.probe_initial_position        
        if self.use_ramp:
            self.push_cmd = {
                "$type": "apply_force_to_object",
                "force": self.push_force,
                "id": int(o_id)
            }
        else:
            self.push_position = {
                k:v+self.force_offset[k]*self.rotate_vector_parallel_to_floor(
                    self.probe_scale, rot['y'])[k]
                for k,v in self.push_position.items()}
            self.push_position = {
                k:v+random.uniform(-self.force_offset_jitter, self.force_offset_jitter)
                for k,v in self.push_position.items()}

            self.push_cmd = {
                "$type": "apply_force_at_position",
                "force": self.push_force,
                "position": self.push_position,
                "id": int(o_id)
            }

        # decide when to apply the force
        self.force_wait = int(random.uniform(self.force_wait_range[0], self.force_wait_range[1]))
        print("force wait", self.force_wait)

        if self.force_wait == 0:
            commands.append(self.push_cmd)

        return commands

    def _place_ramp_under_probe(self) -> List[dict]:

        # ramp params
        self.ramp = random.choice(self.DEFAULT_RAMPS)
        ramp_pos = copy.deepcopy(self.probe_initial_position)
        ramp_pos['y'] += self.zone_scale['y'] if not self.remove_zone else 0.0 # don't intersect w zone
        ramp_rot = self.get_y_rotation([180,180])
        ramp_id = self._get_next_object_id()

        # figure out scale
        r_len, r_height, r_dep = self.get_record_dimensions(self.ramp)
        scale_x = (0.75 * self.collision_axis_length) / r_len
        ramp_scale = arr_to_xyz([scale_x, self.scale_to(r_height, 1.5), 0.75 * scale_x])

        cmds = self.add_ramp(
            record = self.ramp,
            position=ramp_pos,
            rotation=ramp_rot,
            scale=ramp_scale,
            o_id=ramp_id,
            add_data=True)

        # give the ramp a texture and color
        cmds.extend(
            self.get_object_material_commands(
                # self.ramp, ramp_id, self.get_material_name(self.target_material)))
                self.ramp, ramp_id, self.get_material_name("plastic_vinyl_glossy_white")))        
        # rgb = self.random_color(exclude=self.target_color, exclude_range=0.5)
        rgb = [0.75,0.75,1.0]
        cmds.append(
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": ramp_id})            
            

        print("ramp commands")
        print(cmds)

        # need to adjust probe height as a result of ramp placement
        self.probe_initial_position['x'] -= 0.5 * ramp_scale['x'] * r_len - 0.15
        self.probe_initial_position['y'] = ramp_scale['y'] * r_height


        return cmds

    def _replace_target_with_object(self, record, data):
        self.target = record
        self.target_type = data["name"]
        self.target_color = data["color"]
        self.target_scale = data["scale"]
        self.target_id = data["id"]

        self.replace_target = True

    def _build_intermediate_structure(self) -> List[dict]:
        """
        Abstract method for building a physically interesting intermediate structure between the probe and the target.
        """
        commands = []
        return commands

    def _set_distractor_objects(self) -> None:

        self.distractors = OrderedDict()
        for i in range(self.num_distractors):
            record, data = self.random_model(self.distractor_types, add_data=True)
            self.distractors[data['id']] = record

    def _set_occluder_objects(self) -> None:
        self.occluders = OrderedDict()
        for i in range(self.num_occluders):
            record, data = self.random_model(self.occluder_types, add_data=True)
            self.occluders[data['id']] = record

    @staticmethod
    def get_record_dimensions(record: ModelRecord) -> List[float]:
        length = np.abs(record.bounds['left']['x'] - record.bounds['right']['x'])
        height = np.abs(record.bounds['top']['y'] - record.bounds['bottom']['y'])        
        depth = np.abs(record.bounds['front']['z'] - record.bounds['back']['z'])
        return (length, height, depth)

    @staticmethod
    def scale_to(current_scale : float, target_scale : float) -> float:

        return target_scale / current_scale
    
    def _place_background_distractors(self) -> List[dict]:
        """
        Put one or more objects in the background of the scene; they will not interfere with trial dynamics
        """

        commands = []

        # randomly sample distractors and give them obj_ids
        self._set_distractor_objects()

        # distractors will be placed opposite camera
        opposite = np.array([-self.camera_position['x'], 0., -self.camera_position['z']])
        opposite /= np.linalg.norm(opposite)
        opposite = arr_to_xyz(opposite)

        max_theta = 20. * (self.num_distractors - 1) * np.sign(opposite['z'])
        thetas = np.linspace(-max_theta, max_theta, self.num_distractors)
        for i, o_id in enumerate(self.distractors.keys()):
            record = self.distractors[o_id]
            print("distractor record")
            print(record.name, record.wcategory)

            # set a position
            theta = thetas[i]
            pos_unit = self.rotate_vector_parallel_to_floor(opposite, theta)
            d_len, d_height, d_dep = self.get_record_dimensions(record)
            
            pos = self.scale_vector(pos_unit, d_len)
            if i == 0:
                d_len_last = -d_len
                last_x = pos['x']
            x_offset = d_len_last + 0.6*d_len
            pos = arr_to_xyz(
                [min([pos['x'] - x_offset, last_x - x_offset]),
                 0.,
                 np.sign(opposite['z'])*max([d_dep, self.middle_scale['z'] * 4.0])])
            d_len_last = 0.6*d_len
            last_x = pos['x']

            # face toward camera
            ang = 0. if (self.camera_rotation > 0.) else 180.
            rot = self.get_y_rotation([ang, ang])
            
            # add the object
            commands.append(
                self.add_transforms_object(
                    record=record,
                    position=pos,
                    rotation=rot,
                    o_id=o_id,
                    add_data=True))

            # give it a texture if it's a primitive
            if record.name in MODEL_NAMES:
                commands.extend(
                    self.get_object_material_commands(
                        record, o_id, self.get_material_name(self.target_material)))            
            

            # make sure it doesn't have the same color as the target object
            rgb = self.random_color(exclude=self.target_color, exclude_range=0.5)
            scale = arr_to_xyz([1.,1.,1.])
            commands.extend([
                {"$type": "set_color",
                 "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
                 "id": o_id},
                {"$type": "scale_object",
                 "scale_factor": scale,
                 "id": o_id}
            ])

            # todo: give it a random texture if it's a primitive

            # add the metadata
            self.colors = np.concatenate([self.colors, np.array(rgb).reshape((1,3))], axis=0)
            self.scales.append(scale)

        return commands

    def _place_occluders(self) -> List[dict]:
        """
        Put one or more objects in the foreground to occlude the intermediate part of the scene
        """

        commands = []

        # randomly sample occluders and give them obj_ids
        self._set_occluder_objects()

        # path to camera
        camera_ray = np.array([self.camera_position['x'], 0., self.camera_position['z']])
        camera_ray /= np.linalg.norm(camera_ray)
        camera_ray = arr_to_xyz(camera_ray)

        camera_distance = np.linalg.norm(xyz_to_arr(camera_ray))

        max_theta = 20. * (self.num_occluders - 1)
        thetas = np.linspace(-max_theta, max_theta, self.num_occluders)
        for i, o_id in enumerate(self.occluders.keys()):
            record = self.occluders[o_id]

            # set a position
            theta = thetas[i]
            pos_unit = self.rotate_vector_parallel_to_floor(camera_ray, theta)

            o_len, o_height, o_dep = self.get_record_dimensions(record)
            scale = 1. / np.sqrt(o_len**2 + o_dep**2)
            pos = self.scale_vector(pos_unit, 0.75 * camera_distance)

            if i == 0:
                o_len_last = -o_len
                last_x = pos['x']
                
            if self.num_occluders > 1:
                x_offset = o_len_last + 0.6 * o_len
            else:
                x_offset = 0.
                
            pos = arr_to_xyz(
                [min([pos['x'] - x_offset, last_x - x_offset]),
                 0.,
                 np.sign(camera_ray['z']) * max([o_len, o_dep, pos['z'], self.middle_scale['z']*4.0])])
            o_len_last = 0.6*o_len
            last_x = pos['x']

            # face the camera
            ang = self.camera_rotation
            rot = self.get_y_rotation([ang - 30., ang + 30.])

            print("occluder name", record.name)
            print("camera ray", camera_ray)
            print("pos_unit", pos_unit)
            print("pos", pos)

            # add the occluder
            commands.append(
                self.add_transforms_object(
                    record=record,
                    position=pos,
                    rotation=rot,
                    o_id=o_id,
                    add_data=True))

            # give it a texture if it's a primitive
            if record.name in MODEL_NAMES:
                commands.extend(
                    self.get_object_material_commands(
                        record, o_id, self.get_material_name(self.target_material)))            
            
        
            # make sure it doesn't have the same color as the target object
            rgb = self.random_color(exclude=self.target_color, exclude_range=0.5)

            # do some trigonometry to figure out the scale of the occluder
            occ_dist = np.sqrt(pos['x']**2 + pos['z']**2)
            occ_dist *= np.cos(np.radians(theta))
            occ_target_height = self.camera_aim['y'] + occ_dist * np.tan(np.radians(self.camera_altitude))
            occ_target_height *= self.occlusion_scale
            
            scale_y = self.scale_to(o_height, occ_target_height)
            print("scale_y", scale_y)
            scale = arr_to_xyz([scale, scale_y, scale])
            commands.extend([
                {"$type": "set_color",
                 "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
                 "id": o_id},
                {"$type": "scale_object",
                 "scale_factor": scale,
                 "id": o_id}
            ])

            # add the metadata
            self.colors = np.concatenate([self.colors, np.array(rgb).reshape((1,3))], axis=0)
            self.scales.append(scale)

        return commands
        

class MultiDominoes(Dominoes):

    def __init__(self,
                 port: int = 1071,
                 middle_objects=None,
                 num_middle_objects=1,
                 middle_color=None,
                 middle_scale_range=None,
                 middle_rotation_range=None,
                 middle_mass_range=[2.,7.],
                 horizontal=False,
                 spacing_jitter=0.2,
                 lateral_jitter=0.2,
                 middle_material=None,
                 remove_middle=False,
                 **kwargs):

        super().__init__(port=port, **kwargs)

        # Default to same type as target
        self.set_middle_types(middle_objects)

        # Appearance of middle objects
        self.middle_scale_range = middle_scale_range or self.target_scale_range
        self.middle_mass_range = middle_mass_range
        self.middle_rotation_range = middle_rotation_range
        self.middle_color = middle_color
        self.randomize_colors_across_trials = False if (middle_color is not None) else True
        self.middle_material = self.get_material_name(middle_material)
        self.horizontal = horizontal
        self.remove_middle = remove_middle

        # How many middle objects and their spacing
        self.num_middle_objects = num_middle_objects
        self.spacing = self.collision_axis_length / (self.num_middle_objects + 1.)
        self.spacing_jitter = spacing_jitter
        self.lateral_jitter = lateral_jitter

    def set_middle_types(self, olist):
        if olist is None:
            self._middle_types = self._target_types
        else:
            tlist = self.get_types(olist)
            self._middle_types = tlist

    def clear_static_data(self) -> None:
        super().clear_static_data()

        self.middle_type = None
        self.distractors = OrderedDict()
        self.occluders = OrderedDict()
        
        if self.randomize_colors_across_trials:
            self.middle_color = None

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        static_group.create_dataset("remove_middle", data=self.remove_middle)
        static_group.create_dataset("middle_objects", data=[self.middle_type for _ in range(self.num_middle_objects)])        
        if self.middle_type is not None:
            static_group.create_dataset("middle_type", data=self.middle_type)

    @staticmethod
    def get_controller_label_funcs(classname = 'MultiDominoes'):
        funcs = super(MultiDominoes, MultiDominoes).get_controller_label_funcs(classname)

        def num_middle_objects(f):
            try:
                return int(len(f['static']['middle_objects']))
            except KeyError:
                return int(len(f['static']['mass']) - 3)
        def remove_middle(f):
            try:
                return bool(np.array(f['static']['remove_middle']))
            except KeyError:
                return bool(False)

        funcs += [num_middle_objects, remove_middle]
        
        return funcs

    def _build_intermediate_structure(self) -> List[dict]:
        # set the middle object color
        if self.monochrome:
            self.middle_color = self.random_color(exclude=self.target_color)

        return self._place_middle_objects() if bool(self.num_middle_objects) else []

    def _place_middle_objects(self) -> List[dict]:

        offset = -0.5 * self.collision_axis_length
        min_offset = offset + self.target_scale["x"]
        max_offset = 0.5 * self.collision_axis_length - self.target_scale["x"]

        commands = []

        if self.remove_middle:
            rm_idx = random.choice(range(self.num_middle_objects))
        else:
            rm_idx = -1
        
        for m in range(self.num_middle_objects):
            offset += self.spacing * random.uniform(1.-self.spacing_jitter, 1.+self.spacing_jitter)
            offset = np.minimum(np.maximum(offset, min_offset), max_offset)
            if offset >= max_offset:
                print("couldn't place middle object %s" % str(m+1))
                print("offset now", offset)
                break

            if m == rm_idx:
                continue

            record, data = self.random_primitive(self._middle_types,
                                                 scale=self.middle_scale_range,
                                                 color=self.middle_color,
                                                 exclude_color=self.target_color
            )
            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
            zpos = scale["z"] * random.uniform(-self.lateral_jitter, self.lateral_jitter)
            pos = arr_to_xyz([offset, 0., zpos])
            rot = self.get_y_rotation(self.middle_rotation_range)
            if self.horizontal:
                rot["z"] = 90
                pos["z"] += -np.sin(np.radians(rot["y"])) * scale["y"] * 0.5
                pos["x"] += np.cos(np.radians(rot["y"])) * scale["y"] * 0.5
            self.middle_type = data["name"]
            self.middle_scale = {k:max([scale[k], self.middle_scale[k]]) for k in scale.keys()}

            commands.extend(
                self.add_physics_object(
                    record=record,
                    position=pos,
                    rotation=rot,
                    mass=random.uniform(*get_range(self.middle_mass_range)),
                    dynamic_friction=0.5,
                    static_friction=0.5,
                    bounciness=0.,
                    o_id=o_id))

            # Set the middle object material
            commands.extend(
                self.get_object_material_commands(
                    record, o_id, self.get_material_name(self.middle_material)))

            # Scale the object and set its color.
            commands.extend([
                {"$type": "set_color",
                 "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
                 "id": o_id},
                {"$type": "scale_object",
                 "scale_factor": scale,
                 "id": o_id}])

        return commands

if __name__ == "__main__":
    import platform, os
    
    args = get_args("dominoes")
    
    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":0." + str(args.gpu)
        else:
            os.environ["DISPLAY"] = ":0"

    DomC = MultiDominoes(
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
        remove_middle=args.remove_middle
    )

    if bool(args.run):
        DomC.run(num=args.num,
                 output_dir=args.dir,
                 temp_path=args.temp,
                 width=args.width,
                 height=args.height,
                 save_passes=args.save_passes.split(','),
                 save_movies=args.save_movies,
                 save_labels=args.save_labels,
                 args_dict=vars(args))
    else:
        end = DomC.communicate({"$type": "terminate"})
        print([OutputData.get_data_type_id(r) for r in end])
