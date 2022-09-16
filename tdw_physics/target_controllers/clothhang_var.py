from argparse import ArgumentParser
from tdw.output_data import OutputData, Transforms, Images, CameraMatrices
import h5py
import json
import copy
import importlib
import numpy as np
from enum import Enum
import random
import scipy
from scipy.spatial.transform import Rotation as R

from typing import List, Dict, Tuple
from weighted_collection import WeightedCollection
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelRecord
from tdw.librarian import MaterialLibrarian

librarian = MaterialLibrarian()

from tdw_physics.rigidbodies_dataset import (RigidbodiesDataset,
                                             get_random_xyz_transform,
                                             handle_random_transform_args,
                                             get_range)
from tdw_physics.util import MODEL_LIBRARIES, get_parser, xyz_to_arr, arr_to_xyz
from tdw_physics.target_controllers.dominoes_var import Dominoes, MultiDominoes, get_args, none_or_str, none_or_int

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]

CONTAINER_NAMES = [r.name for r in MODEL_LIBRARIES['models_special.json'].records if "fluid_receptacle1x1" in r.name]
OCCLUDER_CATS = "coffee table,houseplant,vase,chair,dog,sofa,flowerpot,coffee maker,stool,laptop,laptop computer,globe,bookshelf,desktop computer,garden plant,garden plant,garden plant"
DISTRACTOR_CATS = "coffee table,houseplant,vase,chair,dog,sofa,flowerpot,coffee maker,stool,laptop,laptop computer,globe,bookshelf,desktop computer,garden plant,garden plant,garden plant"


def get_drop_args(dataset_dir: str):
    """
    Combine Drop-specific arguments with controller-common arguments
    """
    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')

    parser.add_argument("--drop",
                        type=str,
                        default=None,
                        help="comma-separated list of possible drop objects")
    parser.add_argument("--target",
                        type=str,
                        default=None,
                        help="comma-separated list of possible target objects")
    parser.add_argument("--ymin",
                        type=float,
                        default=1.25,
                        help="min height to drop object from")
    parser.add_argument("--ymax",
                        type=float,
                        default=1.5,
                        help="max height to drop object from")
    parser.add_argument("--dscale",
                        type=str,
                        default="[0.1,0.4]",
                        help="scale of drop objects")
    parser.add_argument("--tscale",
                        type=str,
                        default="[0.3,0.7]",
                        help="scale of target objects")
    parser.add_argument("--drot",
                        type=str,
                        default="{'x':[0,360],'y':[0,360],'z':[0,360]}",
                        help="comma separated list of initial drop rotation values")
    parser.add_argument("--zscale",
                        type=str,
                        default="1.2,0.01,1.2",
                        help="scale of target zone")
    parser.add_argument("--zdloc",
                        type=int,
                        default="-1",
                        help="comma-separated list of possible target zone shapes")

    parser.add_argument("--zjitter",
                        type=float,
                        default=0.2,
                        help="amount of z jitter applied to the target zone")


    ### force
    parser.add_argument("--fscale",
                        type=str,
                        default="[5.0,5.0]",
                        help="range of scales to apply to push force")
    parser.add_argument("--frot",
                        type=str,
                        default="[0,0]",
                        help="range of angles in xz plane to apply push force")
    parser.add_argument("--fjitter",
                        type=float,
                        default=0,
                        help="jitter around object centroid to apply force")
    parser.add_argument("--mrot",
                        type=str,
                        default="{'x':[0,360],'y':[0,360],'z':[0,360]}",
                        help="comma separated list of initial middle rotation values")
    parser.add_argument("--jitter",
                        type=float,
                        default=0.2,
                        help="amount to jitter initial drop object horizontal position across trials")
    # parser.add_argument("--camera_distance",
    #                     type=float,
    #                     default=1.25,
    #                     help="radial distance from camera to drop/target object pair")
    # parser.add_argument("--camera_min_angle",
    #                     type=float,
    #                     default=0,
    #                     help="minimum angle of camera rotation around centerpoint")
    # parser.add_argument("--camera_max_angle",
    #                     type=float,
    #                     default=0,
    #                     help="maximum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_min_height",
                        type=float,
                        default=0.,
                         help="min height of camera as a fraction of drop height")
    parser.add_argument("--camera_max_height",
                        type=float,
                        default=2.,
                        help="max height of camera as a fraction of drop height")
    parser.add_argument("--mmass",
                    type=str,
                    default="10.0",
                    help="Scale or scale range for mass of  middle object")
    ### occluder/distractors
    parser.add_argument("--occluder_categories",
                                      type=none_or_str,
                                      default=OCCLUDER_CATS,
                                      help="the category ids to sample occluders from")
    parser.add_argument("--distractor_categories",
                                      type=none_or_str,
                                      default=DISTRACTOR_CATS,
                                      help="the category ids to sample distractors from")


    def postprocess(args):
         # whether to set all objects same color
        args.monochrome = bool(args.monochrome)

        args.dscale = handle_random_transform_args(args.dscale)
        # args.tscale = handle_random_transform_args(args.tscale)

        args.drot = handle_random_transform_args(args.drot)
        # args.trot = handle_random_transform_args(args.trot)

        # choose a valid room
        assert args.room in ['box', 'tdw', 'house'], args.room

        if args.drop is not None:
            drop_list = args.drop.split(',')
            assert all([d in MODEL_NAMES for d in drop_list]), \
                "All drop object names must be elements of %s" % MODEL_NAMES
            args.drop = drop_list
        else:
            args.drop = MODEL_NAMES

        # if args.target is not None:
        #     targ_list = args.target.split(',')
        #     assert all([t in MODEL_NAMES for t in targ_list]), \
        #         "All target object names must be elements of %s" % MODEL_NAMES
        #     args.target = targ_list
        # else:
        #     args.target = MODEL_NAMES

        # if args.color is not None:
        #     rgb = [float(c) for c in args.color.split(',')]
        #     assert len(rgb) == 3, rgb
        #     args.color = rgb

        return args

    args = parser.parse_args()
    args = domino_postproc(args)
    args = postprocess(args)

    return args


class ClothHit(MultiDominoes):
    """
    Drop a random Flex primitive object on another random Flex primitive object
    """

    def __init__(self,
                 port: int = None,
                 drop_objects=MODEL_NAMES,
                 target_objects=MODEL_NAMES,
                 height_range=[0.5, 1.5],
                 drop_scale_range=[0.1, 0.4],
                 target_scale_range=[0.3, 0.6],
                 zone_scale_range={'x':2.,'y':0.01,'z':2.},
                 drop_jitter=0.02,
                 zjitter = 0,
                 zone_dloc = -1,
                 drop_rotation_range=None,
                 target_rotation_range=None,
                 middle_rotation_range=None,
                 middle_mass_range=[10.,11.],
                 middle_scale_range=None,
                 target_color=None,
                 camera_radius=1.0,
                 camera_min_angle=0,
                 camera_max_angle=360,
                 camera_min_height=1./3,
                 camera_max_height=2./3,
                 room = "box",
                 target_zone=['sphere'],
                 zone_location = None,
                 **kwargs):

        ## initializes static data and RNG
        super().__init__(port=port, target_color=target_color, **kwargs)
        self.zjitter = zjitter
        self.room = room
        self.use_obi = True
        self.obi_unique_ids = 0

        self.zone_scale_range = zone_scale_range

        if zone_location is None: zone_location = TDWUtils.VECTOR3_ZERO
        self.zone_location = zone_location

        self.set_zone_types(target_zone)

        ## allowable object types
        self.set_drop_types(drop_objects)
        self.set_target_types(target_objects)
        self._middle_types = self._target_types

        ## object generation properties
        self.height_range = height_range
        self.drop_scale_range = drop_scale_range
        self.target_scale_range = target_scale_range
        self.drop_jitter = drop_jitter
        self.target_color = target_color
        self.drop_rotation_range = drop_rotation_range
        self.target_rotation_range = target_rotation_range
        self.middle_rotation_range = middle_rotation_range
        self.middle_mass_range = middle_mass_range
        self.middle_scale_range = middle_scale_range

        ## camera properties
        self.camera_radius = camera_radius
        self.camera_min_angle = camera_min_angle
        self.camera_max_angle = camera_max_angle
        self.camera_min_height = camera_min_height
        self.camera_max_height = camera_max_height
        self._material_types = ["Fabric", "Leather", "Paper", "Plastic"]
        self.zone_dloc = zone_dloc

        self._candidate_types = self._target_types
        self.candidate_scale_range = self.target_scale_range

        all_material = []
        for mtype in self.material_types:
             all_material += librarian.get_all_materials_of_type(mtype)
        #import ipdb; ipdb.set_trace()
        self.all_material_names  = [m.name for m in all_material if (not m.name.startswith("alum") and not m.name.startswith("metal"))]

        self.force_wait = 10

    # def get_types(self, objlist):
    #     recs = MODEL_LIBRARIES["models_flex.json"].records
    #     tlist = [r for r in recs if r.name in objlist]
    #     return tlist

    def get_scene_initialization_commands(self) -> List[dict]:

        selected_room = ""
        if self.room == 'random':
            selected_room = random.choice(['box', 'tdw', 'mmcraft'])


        if self.room == 'box' or selected_room == "box":
            add_scene = self.get_add_scene(scene_name="box_room_2018")
        elif self.room == 'tdw' or selected_room == "tdw":
            add_scene = self.get_add_scene(scene_name="tdw_room")
        elif self.room == 'house' or selected_room == "house":
            add_scene = self.get_add_scene(scene_name='archviz_house')
        elif self.room == 'mmcraft' or selected_room == "mmcraft":
            add_scene = self.get_add_scene(scene_name='mm_craftroom_1b')
        print("room name", self.room, selected_room)
        commands = [add_scene,
                {"$type": "set_aperture",
                 "aperture": 4.0},
                {"$type": "set_post_exposure",
                 "post_exposure": 0.4},
                {"$type": "set_ambient_occlusion_intensity",
                 "intensity": 0.01}, #0.01
                {"$type": "set_ambient_occlusion_thickness_modifier",
                 "thickness": 3.5},
                 {"$type": "rotate_directional_light_by", "angle": -30, "axis": "yaw"},
                 {"$type": "rotate_directional_light_by", "angle": 30, "axis": "roll"},
                 ]

        if self.room == 'tdw' or selected_room == "tdw":
            commands.extend([
                {"$type": "adjust_directional_light_intensity_by", "intensity": 0.25},
                {"$type": "adjust_point_lights_intensity_by", "intensity": 0.6},
                {"$type": "set_shadow_strength", "strength": 0.5},
                {"$type": "rotate_directional_light_by", "angle": -30, "axis": "pitch", "index": 0},
            ])
        elif self.room == 'box' or selected_room == "box":
            commands.extend([
                {"$type": "adjust_directional_light_intensity_by", "intensity": 0.7},
                {"$type": "set_shadow_strength", "strength": 0.6},
            ])
        elif self.room == 'house' or selected_room == "house":
            commands.extend([
             {"$type": "adjust_directional_light_intensity_by", "intensity": 0.4},
             {"$type": "adjust_point_lights_intensity_by", "intensity": 0.5},
             {"$type": "set_shadow_strength", "strength": 0.8}])
        elif self.room == 'mmcraft' or selected_room == "mmcraft":
            commands.extend([
                {"$type": "adjust_point_lights_intensity_by", "intensity": 0.6},
                {"$type": "adjust_directional_light_intensity_by", "intensity": 0.2},
                {"$type": "set_shadow_strength", "strength": 0.5}])


        return commands

    def get_stframe_pred(self):
        frame_id = self.start_frame_after_curtain  + self.stframe_whole_video + 10
        return frame_id

    def set_drop_types(self, olist):
        tlist = self.get_types(olist)
        self._drop_types = tlist

    def set_target_types(self, olist, libraries=["models_flex.json"]):
        tlist = self.get_types(olist, libraries=libraries, flex_only=self.flex_only)

        self._target_types = tlist

    def clear_static_data(self) -> None:
        super().clear_static_data()

        ## scenario-specific metadata: object types and drop position
        self.heights = np.empty(dtype=np.float32, shape=0)
        self.target_type = None
        self.drop_type = None
        self.drop_position = None
        self.drop_rotation = None
        self.target_rotation = None

    def get_field_of_view(self) -> float:
        return 55


    def generate_static_object_info(self):

        # color for "star object"
        colors = [[0.01844594, 0.77508636, 0.12749255],#pink
                  [0.17443318, 0.22064707, 0.39867442],#black
                  [0.75136046, 0.06584012, 0.22674323],#red
                  [0.47, 0.38,   0.901],#purple
                   ]
        non_star_color = [246/255, 234/255, 224/255]

        self.repeat_trial = False
        # sample distinct objects
        self.candidate_dict = dict()
        self.star_object = dict()
        self.star_object["type"] = random.choice(self._star_types)
        self.star_object["color"] = self.random_color_exclude_list(exclude_list=[[1.0, 0, 0], non_star_color, [1.0, 1.0, 0.0]], hsv_brightness=0.7)
        #colors[distinct_id] #np.array(self.random_color(None, 0.25))[0.9774568,  0.87879388, 0.40082996]#orange
        self.star_object["mass"] =  2000 #0.0002 #2 * 10 ** np.random.uniform(-1,1) #random.choice([0.1, 2.0, 10.0])
        self.star_object["scale"] = get_random_xyz_transform(self.star_scale_range)
        self.star_object["material"] = random.choice(self.all_material_names)
        self.star_object["deform"] = self.var_rng.uniform(0,1)

        #self.star_object["material"] =
        print("====star object mass", self.star_object["mass"])

        self.zone_friction = 0.02
        #distinct_masses = [0.1, 2.0, 10.0]
        mass = 0.04 #0.01
        self.normal_mass = 2.0
        random.shuffle(colors)
        #random.shuffle(distinct_masses)
        ## add the non-star objects have the same weights

        for distinct_id in range(1):
            self.candidate_dict[distinct_id] = dict()
            self.candidate_dict[distinct_id]["type"] = random.choice(self._candidate_types)
            self.candidate_dict[distinct_id]["scale"] = get_random_xyz_transform(self.candidate_scale_range)
            self.candidate_dict[distinct_id]["color"] = non_star_color#[0.9774568,  0.87879388, 0.40082996]
            self.candidate_dict[distinct_id]["mass"] = mass
            self.candidate_dict[distinct_id]["friction"] = 0.2 #0.0001



    def get_trial_initialization_commands(self, interact_id) -> List[dict]:
        commands = []

        # randomization across trials
        # if not(self.randomize):
        #     self.trial_seed = (self.MAX_TRIALS * self.seed) + self._trial_num
        #     random.seed(self.trial_seed)
        # else:
        #     self.trial_seed = -1 # not used

        self.offset = [0, 0]
        # Place target zone
        commands.extend(self._place_target_zone(interact_id))

        # Choose and drop an object.
        commands.extend(self._place_star_object(interact_id))

        # Choose and place a middle object.
        commands.extend(self._place_intermediate_object(interact_id))

        # Teleport the avatar to a reasonable position based on the drop height.
        a_pos = self.get_random_avatar_position(radius_min=self.camera_radius_range[0],
                                                radius_max=self.camera_radius_range[1],
                                                angle_min=self.camera_min_angle,
                                                angle_max=self.camera_max_angle,
                                                y_min=self.camera_min_height,
                                                y_max=self.camera_max_height,
                                                center=TDWUtils.VECTOR3_ZERO)

        cam_aim = {"x": 0, "y": 0.5, "z": 0}
        commands.extend([
            {"$type": "teleport_avatar_to",
             "position": a_pos},
            {"$type": "look_at_position",
             "position": cam_aim},
            {"$type": "set_focus_distance",
             "focus_distance": TDWUtils.get_distance(a_pos, cam_aim)}
        ])

        # Set the camera parameters
        self._set_avatar_attributes(a_pos)

        self.camera_position = a_pos
        self.camera_rotation = np.degrees(np.arctan2(a_pos['z'], a_pos['x']))
        dist = TDWUtils.get_distance(a_pos, self.camera_aim)
        self.camera_altitude = np.degrees(np.arcsin((a_pos['y'] - self.camera_aim['y'])/dist))

        # For distractor placements
        self.middle_scale = self.zone_scale

        # Place distractor objects in the background
        commands.extend(self._place_background_distractors(z_pos_scale=1))

        # Place occluder objects in the background
        commands.extend(self._place_occluders(z_pos_scale=1))

        # test mode colors
        if self.use_test_mode_colors:
            self._set_test_mode_colors(commands)

        return commands



    def _place_target_zone(self, interact_id) -> List[dict]:

        # create a target zone (usually flat, with same texture as room)
        if not self.repeat_trial: # sample from scratch

            record, data = self.random_primitive(self._zone_types,
                                                 scale={'x': 0.25, 'y':0.1, 'z':0.4},
                                                 color=self.zone_color,
                                                 add_data=False
            )
            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
            self.zone = record
            self.zone_type = data["name"]
            self.zone_color = rgb
            self.zone_id = o_id
            self.zone_scale = scale
        else:
            # dry pass to get the obj id counter correct
            record, data = self.random_primitive([self.zone],
                                                 scale=self.zone_scale,
                                                 color=self.zone_color,
                                                 add_data=False
            )
            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
            assert(record == self.zone)
            assert(o_id == self.zone_id)
            assert(self.element_wise_equal(scale, self.zone_scale))
            assert(self.element_wise_equal(scale, self.zone_scale))
            assert(np.array_equal(rgb, self.zone_color))


        if any((s <= 0 for s in scale.values())):
            self.remove_zone = True
            self.scales = self.scales[:-1]
            self.colors = self.colors[:-1]
            self.model_names = self.model_names[:-1]
        self.distinct_ids = np.append(self.distinct_ids, -1)
        # place it just beyond the target object with an effectively immovable mass and high friction

        self.zone_location = self._get_zone_location(scale, islast=interact_id==(self.num_interactions-1))
        commands = []
        commands.extend(
            self.add_primitive(
                record=record,
                position=(self.zone_location),
                rotation=TDWUtils.VECTOR3_ZERO,
                scale=scale,
                material=self.zone_material,
                color=rgb,
                mass=0.02,
                scale_mass=False,
                dynamic_friction=self.zone_friction,
                static_friction=(self.zone_friction),
                bounciness=0,
                o_id=o_id,
                add_data=(not self.remove_zone),
                make_kinematic=False # zone shouldn't move
            ))
        # get rid of it if not using a target object
        if self.remove_zone:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]

        if interact_id >0 :
            self.zone_hold_cmd = {"$type": "teleport_object",
                              "id": o_id,
                              "position": self.zone_location}

            if self.zone_dloc == 2:
                self.zone_hold_force_wait =25
            else:
                self.zone_hold_force_wait = 0
                self.zone_hold_cmd = None
        else:
            self.zone_hold_force_wait = 0

        return commands


    def _get_zone_location(self, scale, islast):
        """Where to place the target zone? Right behind the target object."""
        BUFFER = 0

        if not islast:
            return {
                "x": self.offset[0] + random.uniform(self.collision_axis_length - 1.5, self.collision_axis_length - 1.7),# + 0.5 * self.zone_scale_range['x'] + BUFFER,
                "y": random.uniform(0.5, 0.8) if not self.remove_zone else 10.0,
                "z": self.offset[1] + (- 1.0) * random.uniform(0.8, 1.0) + random.uniform(-self.zjitter,self.zjitter) if not self.remove_zone else 10.0
            }
        else:
            if self.zone_dloc == 2:
                return {
                    "x": self.offset[0] + 0.5,# + 0.5 * self.zone_scale_range['x'] + BUFFER,
                    "y": 2.0 if not self.remove_zone else 10.0,
                    "z": self.offset[1] + random.uniform(-self.zjitter,self.zjitter) if not self.remove_zone else 10.0
                }

            elif self.zone_dloc == 1:
                # zone location at the right boundary
                #random.uniform(self.collision_axis_length - 1.25, self.collision_axis_length-1.1),
                return {
                    "x": self.offset[0] - 0.5,# + 0.5 * self.zone_scale_range['x'] + BUFFER,
                    "y": 0.5 if not self.remove_zone else 10.0,
                    "z": self.offset[1] + 0 +  random.uniform(-self.zjitter,self.zjitter) if not self.remove_zone else 10.0
                }

            else:
                raise ValueError(f"zloc needs to be [1,2,3], but get {self.zone_dloc}")




    def get_additional_command_when_removing_curtain(self, frame=0):

        if frame < 10:
            return [self.hold_cmd] + [self.zone_hold_cmd] if self.zone_hold_cmd else []
        elif frame == 10:
            self.is_push = True
            return [self.push_cmd] if self.push_cmd is not None else [] + [self.zone_hold_cmd] if self.zone_hold_cmd else []
        else:
            return []+ [self.zone_hold_cmd] if self.zone_hold_cmd else []

    def get_per_frame_commands(self, resp: List[bytes], frame: int, force_wait=None) -> List[dict]:
        output = []
        if force_wait == None:

            if (self.force_wait != 0) and frame <= self.force_wait:
                # if self.PRINT:
                #     print("applied %s at time step %d" % (self.hold_cmd, frame))
                # output = [self.hold_cmd]
                if not self.is_push:
                    output = [self.push_cmd] if self.push_cmd is not None else []
            else:
                output = []
        else:
            if (self.force_wait != 0) and frame <= force_wait:
                # if self.PRINT:
                #     print("applied %s at time step %d" % (self.hold_cmd, frame))
                # output = [self.hold_cmd]
                if not self.is_push:
                    output = [self.push_cmd] if self.push_cmd is not None else []
            else:
                output = []
        if frame < self.zone_hold_force_wait:
            output.append(self.zone_hold_cmd)

        return output

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        ## color and scales of primitive objects
        # static_group.create_dataset("target_type", data=self.target_type)
        #static_group.create_dataset("drop_type", data=self.drop_type)
        #static_group.create_dataset("drop_position", data=xyz_to_arr(self.drop_position))
        #static_group.create_dataset("drop_rotation", data=xyz_to_arr(self.drop_rotation))
        # static_group.create_dataset("target_rotation", data=xyz_to_arr(self.target_rotation))



    def _write_class_specific_data(self, static_group: h5py.Group) -> None:
        #variables = static_group.create_group("variables")
        static_group.create_dataset("star_mass", data=self.star_object["mass"])
        static_group.create_dataset("star_deform", data=self.star_object["deform"])
        try:
            static_group.create_dataset("star_type", data=self.star_object["type"])
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("star_size", data=xyz_to_arr(self.star_object["scale"]))
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("zdloc", data=self.zone_dloc)
        except (AttributeError,TypeError):
            pass

    def _write_frame_labels(self,
                            frame_grp: h5py.Group,
                            resp: List[bytes],
                            frame_num: int,
                            sleeping: bool) -> Tuple[h5py.Group, List[bytes], int, bool]:

        labels, resp, frame_num, done = RigidbodiesDataset._write_frame_labels(self, frame_grp, resp, frame_num, sleeping)

        # Whether this trial has a target or zone to track
        has_target = (not self.remove_target) or self.replace_target
        has_zone = not self.remove_zone
        labels.create_dataset("has_target", data=has_target)
        labels.create_dataset("has_zone", data=has_zone)
        if not (has_target or has_zone):
            return labels, resp, frame_num, done

        # Whether target moved from its initial position, and how much


        # Whether target has hit the zone
        if has_target and has_zone:
            c_points, c_normals = self.get_object_target_collision(
                self.target_id, self.zone_id, resp)
            target_zone_contact = bool(len(c_points))
            labels.create_dataset("target_contacting_zone", data=target_zone_contact)

        return labels, resp, frame_num, done

    def _write_frame(self,
                     frames_grp: h5py.Group,
                     resp: List[bytes],
                     frame_num: int) -> \
            Tuple[h5py.Group, h5py.Group, dict, bool]:
        frame, objs, tr, sleeping = super()._write_frame(frames_grp=frames_grp,
                                                         resp=resp,
                                                         frame_num=frame_num)
        # If this is a stable structure, disregard whether anything is actually moving.
        return frame, objs, tr, sleeping and frame_num < 300

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame - self.stframe_whole_video > 200

    def get_rotation(self, rot_range):
        if rot_range is None:
            return {"x": 0,
                    "y": random.uniform(0, 360),
                    "z": 0}
        else:
            return get_random_xyz_transform(rot_range)

    def _place_intermediate_object(self, interact_id) -> List[dict]:
        """
        Place a primitive object at the room center.
        """

        # create a target object
        # XXX TODO: Why is scaling part of random primitives
        # but rotation and translation are not?
        # Consider integrating!

        # add the object
        commands = []


        nonstar_type = self.candidate_dict[0]["type"]
        nonstar_scale = self.candidate_dict[0]["scale"]
        nonstar_mass = self.candidate_dict[0]["mass"]
        nonstar_color = self.candidate_dict[0]["color"]
        nonstar_friction = self.candidate_dict[0]["friction"]

        record, data = self.random_primitive([nonstar_type],
                                         scale=nonstar_scale,
                                         color=nonstar_color,
                                         add_data=False)



        #import ipdb; ipdb.set_trace()
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

        #print("o_id ==== ", o_id)
        record_size = {"x":abs(record.bounds['right']['x'] - record.bounds['left']['x']),

         "y":abs(record.bounds['top']['y'] - record.bounds["bottom"]['y']),
         "z":abs(record.bounds['front']['z'] - record.bounds['back']['z'])}
        nonstar_scale = {"x": scale["x"]/record_size["x"], "y": scale["y"]/record_size["y"], "z": scale["z"]/record_size["z"]}


        length = np.random.uniform(0, 0.3, 1)[0]
        #theta_degree = theta * (1 + middle_id) + np.random.uniform(-20, 20, 1)[0]

        if interact_id == 0:
            self.middle_position = {"x":-1, "y": np.random.uniform(0.25, 0.35), "z": np.random.uniform(-0.1, 0.1)}
        else:
            self.middle_position = {"x":0.6, "y": 1.5, "z": 0}

        self.middle_rotation = self.get_rotation(self.middle_rotation_range)

        commands.extend(self.add_primitive(record=record,
                                    o_id=o_id,
                                    position=self.middle_position,
                                    rotation=self.middle_rotation,
                                    scale=nonstar_scale,
                                    color=rgb,
                                    material = self.target_material,
                                    mass=nonstar_mass,
                                    scale_mass=False,
                                    dynamic_friction=nonstar_friction,
                                    static_friction=nonstar_friction,
                                    bounciness=0.,
                                    add_data=True,
                                    make_kinematic=False))


        self.target_id = o_id


        self.target_type = nonstar_type


        commands.extend([
            # {"$type": "set_object_collision_detection_mode",
            #  "mode": "continuous_speculative",
            #  "id": o_id},
            {"$type": "set_object_drag",
             "id": o_id,
             "drag": 0., "angular_drag": 0.}])

        # Apply a force to the target object
        if interact_id == 0:
            self.push_force = self.get_push_force(
                scale_range=nonstar_mass * np.array(self.force_scale_range),
                angle_range=self.force_angle_range,
                yforce=[0.5,0.5])
        else:
            self.push_force = self.get_push_force(
                scale_range=nonstar_mass * np.array(self.force_scale_range),
                angle_range=[(-1)*x for x in self.force_angle_range],
                yforce=[0.0001,0.0001])

        print("push force", self.push_force)
        #self.push_force = self.rotate_vector_parallel_to_floor(
        #    self.push_force, -self.middle_rotation['y'], degrees=True)

        self.push_position = self.middle_position


        self.push_position = {
            k: v+random.uniform(-self.force_offset_jitter,
                                self.force_offset_jitter)
            for k, v in self.push_position.items()}
        self.push_cmd = {
            "$type": "apply_force_at_position",
            "force": self.push_force,
            "position": self.push_position,
            "id": int(o_id)
        }
        if interact_id > 0:
            self.push_cmd = None
        # decide when to apply the force
        self.force_wait = 1
        self.is_push = False
        #int(random.uniform(
        #    *get_range(self.force_wait_range)))
        print("force wait", self.force_wait)

        # if self.force_wait == 0:
        #    commands.append(self.push_cmd)

        # If this scene won't have a target
        if self.remove_target:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]


        self.hold_cmd = {"$type": "teleport_object",
                          "id": o_id,
                          "position": self.middle_position}


        return commands

    def _place_star_object(self, interact_id) -> List[dict]:
        """
        Position a primitive object at some height and drop it.

        :param record: The object model record.
        :param height: The initial height from which to drop the object.
        :param scale: The scale of the object.


        :return: A list of commands to add the object to the simulation.
        """
        from tdw.obi_data.cloth.cloth_material import CLOTH_MATERIALS
        from tdw.obi_data.cloth.cloth_material import ClothMaterial
        from tdw.obi_data.cloth.tether_particle_group import TetherParticleGroup
        from tdw.obi_data.cloth.tether_type import TetherType
        deform = self.star_object["deform"]
        cloth_material = ClothMaterial(visual_material=self.star_object["material"],
                                       texture_scale={"x": 1, "y": 1},
                                       stretching_scale= 0.9 + (1.05-0.9) * deform,
                                       stretch_compliance=0 + 0.02 * deform,
                                       max_compression= 0 + 0.5 * deform,
                                       max_bending=0 + 0.05 * deform,
                                       bend_compliance=0 + 1.0 * deform,
                                       drag=0.05,
                                       lift=0.05,
                                       visual_smoothness=0,
                                       mass_per_square_meter=0.03 + (0.04-0.03) * deform)


        #material = self.get_material_name(self.target_material)

        # cloth_material = ClothMaterial(visual_material=self.star_object["material"],
        #                        texture_scale={"x": 1, "y": 1},
        #                        stretching_scale=0.9,
        #                        stretch_compliance=0,
        #                        max_compression=0,
        #                        max_bending=0,
        #                        drag=0.05,
        #                        lift=0.05,
        #                        visual_smoothness=0,
        #                        mass_per_square_meter=0.03) #doesn't change much



        #self.target_position = {"x": 0.5, "y": 0.55, "z": 0} #1.2 for soft 1.05 for soft

        # add the object
        commands = []
        if self.target_rotation is None:
            self.target_rotation = self.get_rotation(self.target_rotation_range)

        o_id = self.get_unique_id()

        self.obi_object_ids = np.append(self.obi_object_ids, o_id)
        self.obi_object_type = [(o_id, 'tethered_cloth')]

        if interact_id == 0:
            self.target_position = {"x": 0.5, "y": 1.2, "z": 0} #1.2 for soft, y:0.9
            angle = -90
            self.obi.create_cloth_sheet(cloth_material=cloth_material, #cloth_material_names[run_id],
                                   object_id=o_id,
                                   position=self.target_position,
                                   rotation={"x": 0, "y": 0, "z": angle}, #-20
                                   tether_positions={TetherParticleGroup.north_edge: TetherType(object_id=o_id, is_static=True),
                                   TetherParticleGroup.south_edge: TetherType(object_id=o_id, is_static=True),
                                   TetherParticleGroup.west_edge: TetherType(object_id=o_id, is_static=True),
                                   TetherParticleGroup.east_edge: TetherType(object_id=o_id, is_static=True)})

        else:
            self.target_position = {"x": 0.5, "y": 0.80, "z": 0} #1.2 for soft
            angle = random.uniform(23, 28) if self.zone_dloc == 2 else random.uniform(18,23)
            self.obi.create_cloth_sheet(cloth_material=cloth_material, #cloth_material_names[run_id],
                                   object_id=o_id,
                                   position=self.target_position,
                                   rotation={"x": 0, "y": 0, "z": angle}, #random.uniform(20, 25)
                                   #rotation={"x": 0, "y": 0, "z": random.uniform(-152, -157)} if self.zone_dloc == 2 else  {"x": 0, "y": 0, "z": random.uniform(-150, -155)}, #-155 is good
                                   tether_positions={TetherParticleGroup.west_edge: TetherType(object_id=o_id, is_static=True),
                                                     TetherParticleGroup.east_edge: TetherType(object_id=o_id, is_static=True)})


        #                                         TetherParticleGroup.north_edge: TetherType(object_id=o_id, is_static=True)})
        self.obi_scale_factor = 0.6

        #if interact_id ==0:
        #    self.obi_scale_factor *= (1 + (1-deform) * 0.3)
        self.obi.set_solver(scale_factor= self.obi_scale_factor, substeps=4) #0.7 for soft


        # Create an object to drop.
        commands = []

        # place legs (supporter) for the cloth
        stone_color = {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.}
        dims = [[1,1,1], [1,1,-1], [-1,-1,1],[-1,-1,-1]]#, []

        for dim in dims:
            y = self.target_position['y'] * self.obi_scale_factor + dim[1] * self.obi_scale_factor * np.sin(np.radians(angle))
            record, data = self.random_primitive(
                object_types=self.get_types("cube"),
                scale={"x": 0.05, "y":y, "z": 0.05},
                color=self.random_color(exclude=self.target_color),
            )
            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

            pos = {
                "x": self.target_position['x'] * self.obi_scale_factor + dim[0]  * self.obi_scale_factor * np.cos(np.radians(angle)),
                "y": 0,
                "z": self.target_position['z'] * self.obi_scale_factor + dim[2]  * self.obi_scale_factor ,
            }


            commands.extend(self.add_physics_object(
                record=record,
                position=pos,
                rotation=TDWUtils.VECTOR3_ZERO,
                mass=1000,
                dynamic_friction=1.0,
                static_friction=1.0,
                bounciness=0,
                o_id=o_id,
                add_data=True)
            )

            # Set the middle object material
            commands.extend(
                self.get_object_material_commands(
                    record, o_id, self.get_material_name(self.zone_material)))

            # Scale the object and set its color.
            commands.extend([
                {"$type": "set_color",
                    "color": stone_color,
                    "id": o_id},
                {"$type": "scale_object",
                    "scale_factor": scale,
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

        return commands

if __name__ == "__main__":

    args = get_drop_args("drop")

    import platform, os
    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":" + str(args.gpu + 1)
        else:
            os.environ["DISPLAY"] = ":"

    DC = ClothHit(
        randomize=args.random,
        seed=args.seed,
        phyvar=args.phy_var,
        var_rng_seed=args.var_rng_seed,
        height_range=[args.ymin, args.ymax],
        drop_scale_range=args.dscale,
        drop_jitter=args.jitter,
        drop_rotation_range=args.drot,
        drop_objects=args.drop,
        target_objects=args.target,
        target_scale_range=args.tscale,
        target_rotation_range=args.trot,
        target_color=args.color,
        probe_color = args.pcolor,
        middle_rotation_range=args.mrot,
        middle_scale_range=args.mscale,
        middle_objects=args.middle,
        force_scale_range=args.fscale,
        force_angle_range=args.frot,
        force_offset_jitter=args.fjitter,
        zjitter = args.zjitter,
        zone_dloc = args.zdloc,
        camera_radius=args.camera_distance,
        camera_min_angle=args.camera_min_angle,
        camera_max_angle=args.camera_max_angle,
        camera_min_height=args.camera_min_height,
        camera_max_height=args.camera_max_height,
        monochrome=args.monochrome,
        room=args.room,
        target_material=args.tmaterial,
        target_zone=args.zone,
        zone_location=args.zlocation,
        zone_scale_range = args.zscale,
        probe_material=args.pmaterial,
        zone_material=args.zmaterial,
        zone_color=args.zcolor,
        zone_friction=args.zfriction,
        distractor_types=args.distractor,
        distractor_categories=args.distractor_categories,
        num_distractors=args.num_distractors,
        occluder_types=args.occluder,
        occluder_categories=args.occluder_categories,
        num_occluders=args.num_occluders,
        flex_only=args.only_use_flex_objects,
        no_moving_distractors=args.no_moving_distractors,
        use_test_mode_colors=args.use_test_mode_colors
    )

    if bool(args.run):
        DC.run(num=args.num,
               output_dir=args.dir,
               temp_path=args.temp,
               width=args.width,
               height=args.height,
               write_passes=args.write_passes.split(','),
                save_passes=args.save_passes.split(','),
                save_movies=args.save_movies,
                save_labels=args.save_labels,
                save_meshes=args.save_meshes,
                args_dict=vars(args))
    else:
        end = DC.communicate({"$type": "terminate"})
        print([OutputData.get_data_type_id(r) for r in end])