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


class DropCloth(MultiDominoes):
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
                 zone_dloc = -1,
                 drop_jitter=0.02,
                 drop_rotation_range=None,
                 target_rotation_range=None,
                 middle_rotation_range=None,
                 middle_mass_range=[10.,11.],
                 middle_scale_range=None,
                 target_color=None,
                 camera_min_height=1./3,
                 camera_max_height=2./3,
                 room = "box",
                 target_zone=['sphere'],
                 zone_location = None,
                 **kwargs):

        ## initializes static data and RNG
        super().__init__(port=port, target_color=target_color, **kwargs)
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
        self.camera_min_height = camera_min_height
        self.camera_max_height = camera_max_height
        self._material_types = ["Fabric", "Leather",  "Paper", "Plastic"]
        self.zone_dloc = zone_dloc


        all_material = []
        for mtype in self.material_types:
             all_material += librarian.get_all_materials_of_type(mtype)

        self.all_material_names  = [m.name for m in all_material if (not m.name.startswith("alum") and not m.name.startswith("metal"))]

        self.force_wait = 8

    # def get_types(self, objlist):
    #     recs = MODEL_LIBRARIES["models_flex.json"].records
    #     tlist = [r for r in recs if r.name in objlist]
    #     return tlist

    def get_stframe_pred(self):
        if self.zone_dloc == 1:
            frame_id = self.start_frame_after_curtain  + self.stframe_whole_video + 4

        else:
            frame_id = self.start_frame_after_curtain  + self.stframe_whole_video
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

        self.star_object["deform"] = self.var_rng.uniform(0,1.0)

        #self.star_object["material"] =
        print("====star object mass", self.star_object["mass"])

        #distinct_masses = [0.1, 2.0, 10.0]
        mass = 2.0
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

        self.middle_objects = dict()

    def _write_class_specific_data(self, static_group: h5py.Group) -> None:
        #variables = static_group.create_group("variables")
        static_group.create_dataset("star_mass", data=self.star_object["mass"])
        static_group.create_dataset("star_deform", data=self.star_object["deform"])
        static_group.create_dataset("star_type", data=self.star_object["type"].name)
        static_group.create_dataset("star_size", data=xyz_to_arr(self.star_object["scale"]))
        static_group.create_dataset("zdloc", data=self.zone_dloc)


    def get_trial_initialization_commands(self, interact_id) -> List[dict]:
        commands = []

        # randomization across trials
        if not(self.randomize):
            self.trial_seed = (self.MAX_TRIALS * self.seed) + self._trial_num
            random.seed(self.trial_seed)
        else:
            self.trial_seed = -1 # not used

        # Place target zone
        commands.extend(self._place_target_zone(interact_id))

        # Choose and drop an object.
        commands.extend(self._place_star_object(interact_id))

        # Choose and place a middle object.
        commands.extend(self._place_intermediate_object(interact_id))

        # Teleport the avatar to a reasonable position based on the drop height.


        if interact_id == 0:
            self.a_pos, theta = self.get_random_avatar_position(radius_min=self.camera_radius_range[0],
                                                radius_max=self.camera_radius_range[1],
                                                angle_min=self.camera_min_angle,
                                                angle_max=self.camera_max_angle,
                                                y_min=self.camera_min_height,
                                                y_max=self.camera_max_height,
                                                center=TDWUtils.VECTOR3_ZERO,
                                                reflections=self.camera_left_right_reflections,
                                                return_theta=True)

            # print("camera position", self.a_pos)
            # print("theta", theta * 180 / np.pi)

            # import ipdb; ipdb.set_trace()

        # Set the camera parameters
        self._set_avatar_attributes(self.a_pos)




        commands.extend([
            {"$type": "teleport_avatar_to",
             "position": self.camera_position},
            {"$type": "look_at_position",
             "position": self.camera_aim},
            {"$type": "set_focus_distance",
             "focus_distance": TDWUtils.get_distance(self.a_pos, self.camera_aim)}
        ])


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

    def get_random_avatar_position(self,
                                   radius_min: float,
                                   radius_max: float,
                                   y_min: float,
                                   y_max: float,
                                   center: Dict[str, float],
                                   angle_min: float = 0,
                                   angle_max: float = 360,
                                   reflections: bool = False,
                                   return_theta = False,
                                   ) -> Dict[str, float]:
        """
        :param radius_min: The minimum distance from the center.
        :param radius_max: The maximum distance from the center.
        :param y_min: The minimum y positional coordinate.
        :param y_max: The maximum y positional coordinate.
        :param center: The centerpoint.
        :param angle_min: The minimum angle of rotation around the centerpoint.
        :param angle_max: The maximum angle of rotation around the centerpoint.

        :return: A random position for the avatar around a centerpoint.
        """

        a_r = random.uniform(radius_min, radius_max)
        a_x = center["x"] + a_r
        a_z = center["z"] + a_r

        rnd = random.uniform(0,1)
        bad_range = [20, 45]
        if rnd < (bad_range[0]-angle_min)/(bad_range[0]-angle_min + max(angle_max-bad_range[1],0)):
            theta = np.radians(random.uniform(angle_min, bad_range[0]))

        else:
            theta = np.radians(random.uniform(bad_range[1],  angle_max))

        #theta = np.radians(135) # #bad: -30, -40, -50 p.radians(random.uniform(angle_min, angle_max))
        if reflections:
            theta2 = random.uniform(angle_min+180, angle_max+180)
            theta = random.choice([theta, theta2])

        a_y = random.uniform(y_min, y_max)
        a_x_new = np.cos(theta) * (a_x - center["x"]) - np.sin(theta) * (a_z - center["z"]) + center["x"]
        a_z_new = np.sin(theta) * (a_x - center["x"]) + np.cos(theta) * (a_z - center["z"]) + center["z"]
        a_x = a_x_new
        a_z = a_z_new

        if return_theta:
            return {"x": a_x, "y": a_y, "z": a_z}, theta
        else:
            return {"x": a_x, "y": a_y, "z": a_z}

    def get_additional_command_when_removing_curtain(self, frame=0):

        return []


    def get_per_frame_commands(self, resp: List[bytes], frame: int, force_wait=None) -> List[dict]:

        if force_wait == None:
            if (self.force_wait != 0) and frame <= self.force_wait:
                # if self.PRINT:
                #     print("applied %s at time step %d" % (self.hold_cmd, frame))
                # output = [self.hold_cmd]
                output = []
            else:
                output = []
        else:
            if (self.force_wait != 0) and frame <= force_wait:
                # if self.PRINT:
                #     print("applied %s at time step %d" % (self.hold_cmd, frame))
                # output = [self.hold_cmd]
                output = []
            else:
                output = []

        return output

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        ## color and scales of primitive objects
        # static_group.create_dataset("target_type", data=self.target_type)
        #static_group.create_dataset("drop_type", data=self.drop_type)
        #static_group.create_dataset("drop_position", data=xyz_to_arr(self.drop_position))
        #static_group.create_dataset("drop_rotation", data=xyz_to_arr(self.drop_rotation))
        # static_group.create_dataset("target_rotation", data=xyz_to_arr(self.target_rotation))


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


    def get_object_target_collision(self, obj_id: int, target_id: int, resp: List[bytes]):

        target_is_obi = True if (target_id in  self.obi_object_ids.tolist()) else False
        object_is_obi = True if (obj_id in self.obi_object_ids.tolist()) else False


        actor_pos = dict()
        for actor_id in self.obi.actors:
           actor_pos[actor_id] = self.obi.actors[actor_id].positions * self.obi_scale_factor

        if target_is_obi:
            if target_id not in actor_pos:
                return [],[]
            obi_position = actor_pos[target_id]
        else:
            o_id = target_id

        if object_is_obi:
            if obj_id not in actor_pos:
                return [], []
            obi_position = actor_pos[obj_id]
        else:
            o_id = obj_id

        obj_info = self.bo_dict[o_id]
        obj_posrot = self.tr_dict[o_id]
        obj_vertices, _ = self.object_meshes[o_id]
        obj_scale = xyz_to_arr(self.scales[self.object_ids.tolist().index(o_id)])


        xmin, xmax = min(obj_info['left'][0], obj_info['right'][0]), max(obj_info['left'][0], obj_info['right'][0])
        zmin, zmax = min(obj_info['front'][2], obj_info['back'][2]), max(obj_info['front'][2], obj_info['right'][2])
        pos = obj_posrot["pos"]
        rot = obj_posrot["rot"]
        rotm = np.eye(4)
        rotm[:3, :3] = R.from_quat(rot).as_matrix()
        rotm[:3, 3] = pos

        nv = obj_vertices.shape[0]
        trans_ver = np.matmul(rotm, np.concatenate([obj_vertices * obj_scale, np.ones((nv, 1))], 1).T).T[:,:3]
        min_dist = np.min(scipy.spatial.distance_matrix(obi_position, trans_ver, p=2), axis=1)

        ymax = obj_info['top'][1]
        contact_points = obi_position[min_dist < 0.027] #0.026]
        #print(np.unique(min_dist)[:10])


        #print("number of contact points ", len(contact_points))
        #if len(contact_points) > 0:
        #   import ipdb; ipdb.set_trace()

        return (contact_points, [])



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
        return frame - self.stframe_whole_video > 150

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
        commands = []

        if interact_id == 0:
            record, data = self.random_primitive(self._middle_types,
                                         scale=self.middle_scale_range,
                                         color=self.probe_color)
        else:
            record, data = self.random_primitive([self.middle_objects[0]["record"]],
                                         scale=self.middle_objects[0]["scale"],
                                         color=self.middle_objects[0]["color"])

        #import ipdb; ipdb.set_trace()
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]



        if self.zone_dloc == 1:
            self.middle_position = {"x":0, "y": 0, "z": 0} if interact_id > 0 else {"x":-3.0, "y": 0, "z": -1.0}
        else:
            if interact_id > 0:
                length = np.random.uniform(0.53, 0.57, 1)[0]
                theta_degree = 45 + np.random.uniform(-20, 20, 1)[0]

                self.middle_position = {"x":length * np.cos(np.deg2rad(theta_degree)), "y": 0, "z": length * np.sin(np.deg2rad(theta_degree))}
            else:
                self.middle_position =  {"x":3.0, "y": 0, "z": 0}
        if interact_id == 0:
            scale = {'x': 0.9, "y": 0.8, "z":  0.9} if self.zone_dloc == 1 else {'x': 0.3, "y": 0.5, "z":  0.3}  #0.9, 0.9, 0.9
            record_size = {"x":abs(record.bounds['right']['x'] - record.bounds['left']['x']),
             "y":abs(record.bounds['top']['y'] - record.bounds["bottom"]['y']),
             "z":abs(record.bounds['front']['z'] - record.bounds['back']['z'])}

            scale = {"x": scale["x"]/record_size["x"], "y": scale["y"]/record_size["y"], "z": scale["z"]/record_size["z"]}

        # add the object

        self.middle_rotation = self.get_rotation(self.middle_rotation_range)

        commands.extend(self.add_physics_object(record=record,
                                    o_id=o_id,
                                    position=self.middle_position,
                                    rotation=self.middle_rotation,
                                    mass = 1000,
                                    dynamic_friction=0.5,
                                    static_friction=0.5,
                                    bounciness=0.))
        #Set the object material
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.target_material)))

        # # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 0.1},
             "id": o_id},
           {"$type": "scale_object",
            "scale_factor": scale,
            "id": o_id}])

        if interact_id == 0:
            self.middle_objects[0] = dict()
            self.middle_objects[0]["record"] = record
            self.middle_objects[0]["scale"] = scale
            self.middle_objects[0]["color"] = rgb



        num_middle = 3
        theta = np.random.uniform(40, 120, 1)[0]
        theta_offset = 0
        if self.zone_dloc == 2 and interact_id > 0:
            theta = 90
            theta_offset = 45

        for middle_id in range(num_middle):
            if interact_id == 0:
                if self.zone_dloc == 2:
                    self._middle_types = [r for r in self._middle_types if r.name != "bowl"]

                record, data = self.random_primitive(self._middle_types,
                                             scale=self.middle_scale_range,
                                             color=self.probe_color)
            else:
                record, data = self.random_primitive([self.middle_objects[middle_id + 1]["record"]],
                                    scale=self.middle_objects[middle_id + 1]["scale"],
                                    color=self.middle_objects[middle_id + 1]["color"])

            #import ipdb; ipdb.set_trace()
            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

            #print("o_id ==== ", o_id)

            scale["x"] = 0.3
            scale["z"] = 0.3

            if self.zone_dloc == 2 and interact_id > 0:
                length = np.random.uniform(0.5, 0.5, 1)[0]
            else:
                length = np.random.uniform(0.5, 0.7, 1)[0]
            theta_degree = theta_offset + theta * (1 + middle_id) + np.random.uniform(-20, 20, 1)[0]
            #print(theta_degree, length)
            self.middle_position = {"x":length * np.cos(np.deg2rad(theta_degree)), "y": 0, "z": length * np.sin(np.deg2rad(theta_degree))}

            if interact_id > 0 and self.zone_dloc==1:
                self.middle_position["x"] += 3.0


            self.middle_rotation = self.get_rotation(self.middle_rotation_range)

            commands.extend(self.add_physics_object(record=record,
                                        o_id=o_id,
                                        position=self.middle_position,
                                        rotation=self.middle_rotation,
                                        mass = 1000,
                                        dynamic_friction=0.3,
                                        static_friction=0.3,
                                        bounciness=0.))

            #Set the object material
            commands.extend(
                self.get_object_material_commands(
                    record, o_id, self.get_material_name(self.target_material)))

            # # Scale the object and set its color.
            commands.extend([
                {"$type": "set_color",
                 "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 0.1},
                 "id": o_id},
               {"$type": "scale_object",
                "scale_factor": scale,
                "id": o_id}])

            if interact_id == 0:
                self.middle_objects[middle_id + 1] = dict()
                self.middle_objects[middle_id + 1]["record"] = record
                self.middle_objects[middle_id + 1]["scale"] = scale
                self.middle_objects[middle_id + 1]["color"] = rgb

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


        self.target_position = {"x": 0, "y": 1.5, "z": 0}
        if interact_id > 0:
            if self.zone_dloc == 1:
                self.target_position["y"] = 1.7
            else:
                self.target_position["y"] = 1.3
        # add the object
        commands = []
        if self.target_rotation is None:
            self.target_rotation = self.get_rotation(self.target_rotation_range)

        #o_id = self.get_unique_id()
        o_id = self._get_next_object_id()

        self.obi.create_cloth_sheet(cloth_material=cloth_material, #cloth_material_names[run_id],
                               object_id=o_id,
                               position=self.target_position,
                               rotation={"x": 0, "y": 0, "z": 0})

        self.obi_scale_factor = 0.8
        self.obi.set_solver(scale_factor = self.obi_scale_factor, substeps=2)

        self.target_id = o_id
        self.star_id = o_id

        self.obi_object_ids = np.append(self.obi_object_ids, o_id)
        self.obi_object_type = [(o_id, 'cloth')]

        self.target_type = "cloth"
        self.target = None

        # Create an object to drop.
        commands = []

        return commands

    def _place_target_zone(self, interact_id) -> List[dict]:

        # create a target zone (usually flat, with same texture as room)
        record, data = self.random_primitive(self._zone_types,
                                             scale={'x': 0.2, 'y':0.3, 'z':0.2} if self.zone_dloc==2 else {'x': 0.2, 'y':0.4, 'z':0.2},
                                             color=self.zone_color,
                                             add_data=True
        )

        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        self.zone = record
        self.zone_type = data["name"]
        self.zone_color = rgb
        self.zone_id = o_id
        self.zone_scale = scale
        self.zone_mass = 10.0
        # self.zone_location = TDWUtils.VECTOR3_ZERO

        if any((s <= 0 for s in scale.values())):
            self.remove_zone = True
            self.scales = self.scales[:-1]
            self.colors = self.colors[:-1]
            self.model_names = self.model_names[:-1]

        zone_location = self.zone_location
        if interact_id == 0:
            zone_location['x'] = 0.8
            zone_location['z'] = -0.8

        if interact_id > 0:
            if self.zone_dloc == 1:
                length = 0.72
                angle = 30
                #{"x":length * np.cos(np.deg2rad(theta_degree)), "y": 0, "z": length * np.sin(np.deg2rad(theta_degree))}
                zone_location['x'] = length * np.cos(np.deg2rad(angle)) #0.6 #0.6
                zone_location['z'] = length * np.sin(np.deg2rad(angle)) #0.6 #0.6
                zone_location['y'] = random.uniform(1.9, 2.1)
            elif self.zone_dloc == 2:
                zone_location['x'] = random.uniform(-0.2, 0.2) #0.6
                zone_location['z'] = random.uniform(-0.2, 0.2)  #0.6
                zone_location['y'] = 0 #random.uniform(1.9, 2.1)
            else:
                raise ValueError
        # place it just beyond the target object with an effectively immovable mass and high friction
        commands = []
        commands.extend(
            self.add_physics_object(
                record=record,
                position=zone_location,
                rotation=TDWUtils.VECTOR3_ZERO,
                mass=self.zone_mass,
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

        # get rid of it if not using a target object
        if self.remove_zone:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]

        # self.push_cmd = None
        # if interact_id > 0:
        #     self.push_force = self.get_push_force(
        #         scale_range= self.zone_mass * np.array(self.force_scale_range),
        #         angle_range=self.force_angle_range,
        #         yforce=[0,0])

        #     print("push force", self.push_force)
        #     self.push_force = self.rotate_vector_parallel_to_floor(
        #         self.push_force, 0, degrees=True)

        #     self.push_position = zone_location
        #     self.push_position = {
        #         k:v+self.force_offset[k]*self.rotate_vector_parallel_to_floor(
        #             scale, 0)[k]
        #         for k,v in self.push_position.items()}
        #     self.push_position = {
        #         k:v+random.uniform(-self.force_offset_jitter, self.force_offset_jitter)
        #         for k,v in self.push_position.items()}


        #     print("push position", self.push_position)

        #     self.push_cmd = {
        #         "$type": "apply_force_at_position",
        #         "force": self.push_force,
        #         "position": self.push_position,
        #         "id": int(o_id)
        #     }
        #     import ipdb; ipdb.set_trace()

        #     self.zone_force_wait = 10


        return commands

if __name__ == "__main__":

    args = get_drop_args("drop")

    import platform, os
    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":" + str(args.gpu+1)
        else:
            os.environ["DISPLAY"] = ":"

    DC = DropCloth(
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
        zone_dloc = args.zdloc,
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
