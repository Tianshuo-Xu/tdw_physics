from argparse import ArgumentParser
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
from collections import OrderedDict
from weighted_collection import WeightedCollection
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelRecord, MaterialLibrarian
from tdw.librarian import MaterialLibrarian
librarian = MaterialLibrarian()

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
OCCLUDER_CATS = "coffee table,houseplant,vase,chair,dog,sofa,flowerpot,coffee maker,stool,laptop,laptop computer,globe,bookshelf,desktop computer,garden plant,garden plant,garden plant"
DISTRACTOR_CATS = "coffee table,houseplant,vase,chair,dog,sofa,flowerpot,coffee maker,stool,laptop,laptop computer,globe,bookshelf,desktop computer,garden plant,garden plant,garden plant"

def get_rolling_sliding_args(dataset_dir: str, parse=True):

    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')
    ## Changed defaults
    ### zone


    parser.add_argument("--zscale",
                        type=str,
                        default="2.0,0.01,2.0",
                        help="scale of target zone")

    parser.add_argument("--zone",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible target zone shapes")

    parser.add_argument("--zjitter",
                        type=float,
                        default=0.,
                        help="amount of z jitter applied to the target zone")
    parser.add_argument("--zdloc",
                        type=int,
                        default="-1",
                        help="comma-separated list of possible target zone shapes")


    parser.add_argument("--pheight",
                        type=float,
                        default=0.7,
                        help="amount of z jitter applied to the target zone")

    parser.add_argument("--csize",
                        type=float,
                        default=0.6,
                        help="amount of z jitter applied to the target zone")

    ### force
    parser.add_argument("--fscale",
                        type=str,
                        default="[0.0,0.0]",
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
                        default=0.,
                        help="jitter around object centroid to apply force")

    parser.add_argument("--fupforce",
                        type=str,
                        default='[0,0]',
                        help="Upwards component of force applied, with 0 being purely horizontal force and 1 being the same force being applied horizontally applied vertically")


    parser.add_argument("--is_single_ramp",
                        type=float,
                        default=0.,
                        help="create single ramp scenario-specific")

    parser.add_argument("--add_second_ramp",
                        type=float,
                        default=1,
                        help="create stacking ramp scene with 2 ramps")



    ###target
    parser.add_argument("--target",
                        type=str,
                        default="pipe,cube,pentagon,sphere",
                        help="comma-separated list of possible target objects")

    parser.add_argument("--tscale",
                        type=str,
                        default="0.25,0.25,0.25",
                        help="scale of target objects")

    parser.add_argument("--tlift",
                        type=float,
                        default=0.,
                        help="Lift the target object off the floor/ramp. Useful for rotated objects")

    ### layout
    parser.add_argument("--rolling_sliding_axis_length",
                        type=float,
                        default=1.15,
                        help="Length of spacing between target object and zone.")

    ### ramp
    parser.add_argument("--ramp_scale",
                        type=str,
                        default="[0.2,0.25,0.5]",
                        help="Scaling factor of the ramp in xyz.")

    ### ledge
    parser.add_argument("--use_ledge",
                        type=int,
                        default=0,
                        help="Whether to place ledge between the target and the zone")

    parser.add_argument("--ledge",
                        type=str,
                        default="sphere",
                        help="comma-separated list of possible ledge objects")

    parser.add_argument("--ledge_position",
                        type=float,
                        default=0.5,
                        help="Fraction between 0 and 1 where to place the ledge on the axis")

    parser.add_argument("--ledge_scale",
                        type=str,
                        default="[0.05,0.05,100.0]",
                        help="Scaling factor of the ledge in xyz.")

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
        args.fupforce = handle_random_transform_args(args.fupforce)
        args.ramp_scale = handle_random_transform_args(args.ramp_scale)

        ### ledge
        args.use_ledge = bool(args.use_ledge)

        if args.ledge is not None:
            targ_list = args.ledge.split(',')
            assert all([t in MODEL_NAMES for t in targ_list]), \
                "All ledge object names must be elements of %s" % MODEL_NAMES
            args.ledge = targ_list
        else:
            args.ledge = MODEL_NAMES

        args.ledge_scale = handle_random_transform_args(args.ledge_scale)

        return args

    args = parser.parse_args()
    args = domino_postproc(args)
    args = postprocess(args)

    return args

class FricRamp(MultiDominoes):

    def __init__(self,
                 port: int = None,
                 zjitter = 0,
                 zone_dloc = -1,
                 platform_height=0.7,
                 cloth_size=0.6,
                 fupforce = [0.,0.],
                 ramp_scale = [0.2,0.25,0.5],
                 rolling_sliding_axis_length = 1.15,
                 use_ramp = True,
                 use_ledge = False,
                 ledge = ['cube'],
                 ledge_position = 0.5,
                 ledge_scale = [100,0.1,0.1],
                 is_single_ramp = False,
                #  ledge_color = None,
                target_lift = 0,

                 **kwargs):
        # initialize everything in common w / Multidominoes
        super().__init__(port=port, **kwargs)
        self.zjitter = zjitter
        self.fupforce = fupforce
        self.use_ramp = use_ramp
        self.ramp_scale = ramp_scale
        self.rolling_sliding_axis_length = self.collision_axis_length = rolling_sliding_axis_length
        self.use_ledge = use_ledge
        self.ledge = ledge
        self.ledge_position = ledge_position
        self.ledge_scale = ledge_scale
        self.use_obi = True
        self.cloth_size = cloth_size
        # self.ledge_color = ledge_color
        #['ramp_with_platform_30', 'ramp_with_platform_60', 'ramp_with_platform_weld']
        # triangular_prism
        self.DEFAULT_RAMPS = [MODEL_LIBRARIES["models_flex.json"].get_record("triangular_prism")]
        #[MODEL_LIBRARIES["models_full.json"].get_record("ramp_with_platform_weld")]
        self.target_lift = target_lift



        """
        How by what factor should the y and z dimensions should be smaller than the original ramp
        """
        self.second_ramp_factor = 0.5
        self.camera_aim = {"x": 0.3, "y": 1.1, "z": 0.} # fixed aim
        """
        The color of the second ramp
        """
        self.second_ramp_color = [1., 1., 1.] #White

        """
        Vedang's janky way of randomizing the friction and the size of the ramp
        """
        self.zone_dloc = zone_dloc

        self.fric_range= [0.01, 0.2] # 0.20] #[0.01, 0.25] #[0.01, 0.2] #0.65] #0.01 ~ 0.2
        #self.ramp_y_range=[0.9, 0.9]

        self.is_single_ramp = is_single_ramp
        self.num_interactions = 1


        if self.is_single_ramp:
            self.ramp_y_range=[platform_height, platform_height]
            self.add_second_ramp = False
            self.ramp_base_height_range = 0.1
            self.ramp_base_height = random.uniform(*get_range(self.ramp_base_height_range))
            self.camera_aim = {"x": 1.3, "y": 1.1, "z": 0.}
        else:
            self.ramp_y_range=[platform_height, platform_height]#[0.9, 0.9]
            self.add_second_ramp = True
            self.ramp_base_height_range = self.ramp_scale['y']
            self.ramp_base_height = random.uniform(*get_range(self.ramp_base_height_range))

        self._material_types = ["Fabric", "Leather", "Paper", "Plastic"]


        all_material = []
        for mtype in self.material_types:
             all_material += librarian.get_all_materials_of_type(mtype)
        #import ipdb; ipdb.set_trace()
        #exclude = ['aluminium_wire_knit_grill']
        self.all_material_names  = [m.name for m in all_material if (not m.name.startswith("alum") and not m.name.startswith("metal"))]
        self.start_frame_for_prediction = 150


    def _add_ramp_base_to_ramp(self, color=None, sample_ramp_base_height=True, pos=None, is_kinematic=True) -> None:

        cmds = []


        if color is None:
            color = self.random_color(exclude=self.target_color)
        if sample_ramp_base_height:
            self.ramp_base_height = random.uniform(*get_range(self.ramp_base_height_range))
        if self.ramp_base_height < 0.01:
            self.ramp_base_scale = copy.deepcopy(self.ramp_scale)
            return []

        self.ramp_base = self.CUBE
        r_len, r_height, r_dep = self.get_record_dimensions(self.ramp)

        self.ramp_base_scale = arr_to_xyz([
            float(6 * self.ramp_scale['x'] * r_len),
            float(self.ramp_base_height),
            float(0.15 * self.ramp_scale['z'] * r_dep)])
        self.ramp_base_id = self._get_next_object_id()

        # add the base
        ramp_base_physics_info = {
            'mass': 500,
            'dynamic_friction': 0.01,
            'static_friction': 0.01,
            'bounciness': 0}
        if self.ramp_physics_info.get('dynamic_friction', None) is not None:
            ramp_base_physics_info.update(self.ramp_physics_info)

        cmds.extend(
            RigidbodiesDataset.add_physics_object(
                self,
                record=self.ramp_base,
                position=copy.deepcopy(self.ramp_pos) if pos is None else pos,
                rotation=TDWUtils.VECTOR3_ZERO,
                o_id=self.ramp_base_id,
                add_data=True,
                **ramp_base_physics_info))

        # scale it, color it, fix it
        cmds.extend(
            self.get_object_material_commands(
                self.ramp_base, self.ramp_base_id, self.get_material_name(self.ramp_material)))
        cmds.extend([
            {"$type": "scale_object",
             "scale_factor": self.ramp_base_scale,
             "id": self.ramp_base_id},
            {"$type": "set_color",
             "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.},
             "id": self.ramp_base_id},
            {"$type": "set_object_collision_detection_mode",
             "mode": "continuous_speculative",
             "id": self.ramp_base_id},
            {"$type": "set_kinematic_state",
             "id": self.ramp_base_id,
             "is_kinematic": is_kinematic,
             "use_gravity": True}])

        # add data
        self.model_names.append(self.ramp_base.name)
        self.scales.append(self.ramp_base_scale)
        self.colors = np.concatenate([self.colors, np.array(color).reshape((1,3))], axis=0)

        # raise the ramp
        if not self.is_single_ramp:
            self.ramp_pos['y'] += self.ramp_base_scale['y']

        return cmds



    def _place_ramp_under_probe(self) -> List[dict]:

        cmds = []
        if self.is_single_ramp:
            self.ramp = random.choice(self.DEFAULT_RAMPS)
            rgb = self.ramp_color or self.random_color(exclude=self.target_color)
            ramp_pos = copy.deepcopy(self.probe_initial_position)
            ramp_pos['y'] = 0.0 #self.zone_scale['y'] if not self.remove_zone else 0.0 # don't intersect w zone
            ramp_rot = self.get_y_rotation([90,90])

            self.ramp_pos = ramp_pos

            self.ramp_rot = ramp_rot
            # figure out scale
            r_len, r_height, r_dep = self.get_record_dimensions(self.ramp)
            scale_x = (0.75 * self.collision_axis_length) / r_len
            if self.ramp_scale is None:
                self.ramp_scale = arr_to_xyz([scale_x, self.scale_to(r_height, 1.5), 0.75 * scale_x])
            self.ramp_end_x = self.ramp

            second_ramp_pos = copy.deepcopy(self.ramp_pos)
            second_ramp_pos['y'] = 0 #self.zone_scale['y']*1.1 if not self.remove_zone else 0.0 # don't intersect w zone
            rgb = self.ramp_color or self.random_color(exclude=self.target_color)

            ramp_rot = self.get_y_rotation([90,90])
            self.ramp_rot = ramp_rot

            second_ramp_id = self._get_next_object_id()

            self.ramp_id = second_ramp_id
            scale_x = (0.75 * self.collision_axis_length) / r_len
            if self.ramp_scale is None:
                self.ramp_scale = arr_to_xyz([scale_x, self.scale_to(r_height, 1.5), 0.75 * scale_x])
            second_ramp_scale = copy.deepcopy(self.ramp_scale)
            second_ramp_scale['x'] = self.ramp_scale['x']
            second_ramp_scale['y'] = 2 * self.ramp_scale['y']
            second_ramp_scale['z'] = 2 * self.ramp_scale['z']

            second_ramp_pos['x']  -= 0.7 #  - second_ramp_scale['z']

            cmds.extend(
            self.add_ramp(
                record = self.ramp,
                position=second_ramp_pos,
                rotation=self.ramp_rot,
                scale=second_ramp_scale,
                material=self.ramp_material,
                color=rgb,#vedang
                o_id=second_ramp_id,
                add_data=True,
                **self.ramp_physics_info
            ))

            cmds.extend(self._add_ramp_base_to_ramp(color=self.second_ramp_color, pos={'x': 2.8, "y":0, "z":0}))
            self.probe_initial_position['x'] += self.ramp_scale['z']*0.1 - 0.2 #random.uniform(1.0, 1.3) #1.3
            self.probe_initial_position['y'] = self.ramp_scale['y'] * r_height + self.ramp_base_height + self.probe_initial_position['y'] + 1.6

        else:
            # ramp params
            self.ramp = random.choice(self.DEFAULT_RAMPS)
            rgb = self.ramp_color or self.random_color(exclude=self.target_color)
            ramp_pos = copy.deepcopy(self.probe_initial_position)
            ramp_pos['y'] = self.zone_scale['y'] if not self.remove_zone else 0.0 # don't intersect w zone
            ramp_rot = self.get_y_rotation([90,90])

            ramp_id = self._get_next_object_id()

            self.ramp_pos = ramp_pos

            self.ramp_rot = ramp_rot
            self.ramp_id = ramp_id

            # figure out scale
            r_len, r_height, r_dep = self.get_record_dimensions(self.ramp)
            scale_x = (0.75 * self.collision_axis_length) / r_len
            if self.ramp_scale is None:
                self.ramp_scale = arr_to_xyz([scale_x, self.scale_to(r_height, 1.5), 0.75 * scale_x])
            self.ramp_end_x = self.ramp_pos['x'] + self.ramp_scale['x'] * r_len * 0.5

            # optionally add base
            self.ramp_base_height_range = self.ramp_scale['y']
            cmds.extend(self._add_ramp_base_to_ramp(color=self.second_ramp_color))
            # self.ramp_base_height = random.uniform(*get_range(self.ramp_base_height_range))
            ramp_friction = self.star_object["friction"]
            self.ramp_physics_info = {
                    'mass': 500,
                    'dynamic_friction': ramp_friction, #self.zone_friction,
                    'static_friction': ramp_friction, #self.zone_friction,
                    'bounciness': 0}
            # add the ramp
            cmds.extend(
                self.add_ramp(
                    record = self.ramp,
                    position=self.ramp_pos,
                    rotation=self.ramp_rot,
                    scale=self.ramp_scale,
                    material=self.ramp_material,
                    color=rgb,#vedang
                    o_id=self.ramp_id,
                    add_data=True,
                    **self.ramp_physics_info
                ))

            # need to adjust probe height as a result of ramp placement


            self.probe_initial_position['x'] += self.ramp_scale['z']*0.1 - random.uniform(0.8, 1.1)
            self.probe_initial_position['y'] = self.ramp_scale['y'] * r_height + self.ramp_base_height + self.probe_initial_position['y'] + 1.2

            if self.add_second_ramp:
                second_ramp_pos = copy.deepcopy(self.ramp_pos)
                second_ramp_pos['y'] = self.zone_scale['y']*1.1 if not self.remove_zone else 0.0 # don't intersect w zone


                second_ramp_id = self._get_next_object_id()
                scale_x = (0.75 * self.collision_axis_length) / r_len
                if self.ramp_scale is None:
                    self.ramp_scale = arr_to_xyz([scale_x, self.scale_to(r_height, 1.5), 0.75 * scale_x])
                second_ramp_scale = copy.deepcopy(self.ramp_scale)
                second_ramp_scale['x'] = self.ramp_scale['x'] * 0.3
                second_ramp_scale['y'] = self.second_ramp_factor*self.ramp_scale['y'] * 1.0
                second_ramp_scale['z'] = self.second_ramp_factor*self.ramp_scale['z'] * 1.0

                second_ramp_pos['x']  = second_ramp_scale['z']/2. - 0.4

                ramp_friction  = self.star_object["friction"]
                self.second_ramp_physics_info = {
                    'mass': 500,
                    'dynamic_friction': ramp_friction, #self.zone_friction,
                    'static_friction': ramp_friction, #self.zone_friction,
                    'bounciness': 0}

                cmds.extend(
                self.add_ramp(
                    record = self.ramp,
                    position=second_ramp_pos,
                    rotation=self.ramp_rot,
                    scale=second_ramp_scale,
                    material=self.ramp_material,
                    color=rgb,#vedang
                    o_id=second_ramp_id,
                    add_data=True,
                    **self.second_ramp_physics_info
                ))


        return cmds

    def get_trial_initialization_commands(self) -> List[dict]:
        """This is where we string together the important commands of the controller in order"""
        commands = []
        # randomization across trials
        if not(self.randomize):
            self.trial_seed = (self.MAX_TRIALS * self.seed) + self._trial_num
            random.seed(self.trial_seed)
        else:
            self.trial_seed = -1 # not used

        """
        Vedang's janky way of randomizing
        """
        self.zone_friction = random.uniform(*self.fric_range)
        self.ramp_scale['y'] = random.uniform(*self.ramp_y_range)
        print("zone_friction", self.zone_friction)
        print("ramp_scale", self.ramp_scale)
        self.star_object = dict()
        self.star_object["friction"] = self.zone_friction


        # Choose and place the target zone.
        commands.extend(self._place_target_zone())


        # Choose and place a target object.
        commands.extend(self._place_and_push_target_object())

        # Place ledge between target and zone
        if self.use_ledge: commands.extend(self._place_ledge())

        # # Set the probe color
        # if self.probe_color is None:
        #     self.probe_color = self.target_color if (self.monochrome and self.match_probe_and_target_color) else None

        # # Choose, place, and push a probe object.
        # commands.extend(self._place_and_push_probe_object())

        # # Build the intermediate structure that captures some aspect of "intuitive physics."
        # commands.extend(self._build_intermediate_structure())

        # Teleport the avatar to a reasonable position
        a_pos = self.get_random_avatar_position(radius_min=self.camera_radius_range[0],
                                                radius_max=self.camera_radius_range[1],
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

        # Set the camera parameters
        self._set_avatar_attributes(a_pos)

        self.camera_position = a_pos
        self.camera_rotation = np.degrees(np.arctan2(a_pos['z'], a_pos['x']))
        dist = TDWUtils.get_distance(a_pos, self.camera_aim)
        self.camera_altitude = np.degrees(np.arcsin((a_pos['y'] - self.camera_aim['y'])/dist))

        # Place distractor objects in the background
        commands.extend(self._place_background_distractors())

        # Place occluder objects in the background
        commands.extend(self._place_occluders())

        # test mode colors
        if self.use_test_mode_colors:
            self._set_test_mode_colors(commands)

        return commands

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

        if not self.is_single_ramp :
            if self.zone_type == "cube":
                xmin, xmax = min(obj_info['left'][0], obj_info['right'][0]), max(obj_info['left'][0], obj_info['right'][0])
                zmin, zmax = min(obj_info['front'][2], obj_info['back'][2]), max(obj_info['front'][2], obj_info['right'][2])
                ymax = obj_info['top'][1]
                contact_points = obi_position[(obi_position[:,0] > xmin) * (obi_position[:,0] < xmax)]
                contact_points = contact_points[(contact_points[:,2] > zmin) * (contact_points[:,2] < zmax)]
                #print(ymax)
                #print(np.unique(contact_points[:,1])[:10])
                contact_points = contact_points[(contact_points[:,1] - ymax) < 0.02]

            else:
                raise ValueError
        else:
            if self.zone_type == "cube":
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
                contact_points = obi_position[min_dist < 0.026]
                #print(np.unique(min_dist)[:10])
                #contact_points = contact_points[(contact_points[:,1] - ymax) < 0.02]

            else:
                raise ValueError

        #print("number of contact points ", len(contact_points))
        #if len(contact_points) > 0:
        #    import ipdb; ipdb.set_trace()

        return (contact_points, [])


    def _place_and_push_target_object(self) -> List[dict]:
        """
        Place a probe object at the other end of the collision axis, then apply a force to push it.

        Using probe mass and location here to allow for ramps. There is no dedicated probe in this controller.
        """
        # create a target object

        # record, data = self.random_primitive(self._target_types,
        #                                      scale=self.target_scale_range,
        #                                      color=self.target_color,
        #                                      add_data=False)
        #                                      # add_data=(not self.remove_target)
        # o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        # self.target = record
        # self.target_type = data["name"]
        # self.target_color = rgb
        # self.target_scale = self.middle_scale = scale
        # self.target_id = o_id

        # # Where to put the target
        # if self.target_rotation is None:
        #     self.target_rotation = self.get_rotation(self.target_rotation_range)

        # Add the object with random physics values
        commands = []

        ### TODO: better sampling of random physics values
        self.probe_mass = random.uniform(self.probe_mass_range[0], self.probe_mass_range[1])
        if self.is_single_ramp:
            self.probe_initial_position = {"x": -self.rolling_sliding_axis_length + 0.1, "y": self.target_lift - 1.0 + 0.25*(6.0 - self.cloth_size), "z": 0.} #0.1
        else:
            self.probe_initial_position = {"x": -self.rolling_sliding_axis_length + 0.1, "y": self.target_lift, "z": 0.} #0.
        rot = self.get_rotation(self.target_rotation_range)


        self.target_scale = self.middle_scale = {"x": 1, "y": 1, "z":1}

        #tune_param
        #vedang: 'dynamic_friction': 0.0001,
        if self.use_ramp:
            self.ramp_physics_info = {
                'mass': 500,
                'dynamic_friction': self.zone_friction,
                'static_friction': self.zone_friction,
                'bounciness': 0}

            self.ramp_color = [1, 1 ,1]

            commands.extend(self._place_ramp_under_probe())



        from tdw.obi_data.cloth.cloth_material import CLOTH_MATERIALS
        from tdw.obi_data.cloth.cloth_material import ClothMaterial

        visual_material = random.choice(self.all_material_names)
        if self.is_single_ramp:
            cloth_material = ClothMaterial(visual_material,
                               texture_scale={"x": 0.8, "y": 0.8},
                               stretching_scale=1.1, #1.0,
                               stretch_compliance=0.001, #0,
                               max_compression=0.25,
                               max_bending=0.03, #0.01
                               bend_compliance=1.0, #0,
                               drag=0.05,
                               lift=0.05,
                               visual_smoothness=0,
                               mass_per_square_meter=0.15)
        else:
            cloth_material = ClothMaterial(visual_material,
                                   texture_scale={"x": 0.8, "y": 0.8},
                                   stretching_scale=1.1,
                                   stretch_compliance=0.001,
                                   max_compression=0.25,
                                   max_bending=0.03,
                                   bend_compliance=1.0,
                                   drag=0.05,
                                   lift=0.05,
                                   visual_smoothness=0,
                                   mass_per_square_meter=0.15)
            # cloth_material = ClothMaterial(visual_material,
            #                        texture_scale={"x": 0.8, "y": 0.8},
            #                        stretching_scale=0.87,
            #                        stretch_compliance=0.001,
            #                        max_compression=0.25,
            #                        max_bending=0.01,
            #                        bend_compliance=1.0,
            #                        drag=0.05,
            #                        lift=0.05,
            #                        visual_smoothness=0,
            #                        mass_per_square_meter=0.15)



        self.cloth_material = visual_material

        o_id = self._get_next_object_id()
        self.target_id = o_id
        self.star_id = o_id
        self.obi_object_ids = np.append(self.obi_object_ids, o_id)
        self.obi_object_type = [(o_id, 'cloth')]
        #self.object_ids = np.append(self.object_ids, o_id)

        self.obi.create_cloth_sheet(cloth_material=cloth_material, #cloth_material_names[run_id],
                               object_id=o_id,
                               position=self.probe_initial_position,
                               rotation={"x": 0, "y": 0, "z": 0})
        self.obi_scale_factor = self.cloth_size
        self.obi.set_solver(scale_factor=self.obi_scale_factor, substeps=2)

            # self.probe_initial_position['x'] += 0.9*self.target_lift #HACK rotation might've led to the object falling of the back of the ramp, so we're moving it forward

        # the target is the probe
        self.target_position = self.probe_initial_position
        # Scale the object and set its color.
        # commands.extend([
        #     {"$type": "set_color",
        #      "color": {"r": rgb[0], "g": rgb[1clothclothcloth], "b": rgb[2], "a": 1.},
        #      "id": o_id},
            # {"$type": "scale_object",
            #  "scale_factor": scale,
            #  "id": o_id}])

        # Set its collision mode
        # commands.extend([
        #     # {"$type": "set_object_collision_detection_mode",
        #     #  "mode": "continuous_speculative",
        #     #  "id": o_id},
        #     {"$type": "set_object_drag",
        #      "id": o_id,
        #      "drag": 0, "angular_drag": 0}])


        # Apply a force to the target object
        self.push_force = self.get_push_force(
            scale_range=self.probe_mass * np.array(self.force_scale_range),
            angle_range=self.force_angle_range,
            yforce=self.fupforce)
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
                    self.target_scale, rot['y'])[k]
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
        self.push_cmd = {}
        # decide when to apply the force
        self.force_wait = int(random.uniform(*get_range(self.force_wait_range)))
        print("force wait", self.force_wait)


        return commands


    def _place_ledge(self) -> List[dict]:
        """Places the ledge on a location on the axis and fixes it in place"""
        assert self.use_ledge, "need to use ledge"
        commands = []
        self.random_color(exclude=self.target_color)
        record, data = self.random_primitive(
                                                object_types= self.get_types(self.ledge),
                                                 scale=self.ledge_scale,
                                                 color=self.random_color(exclude=self.target_color),
            )
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

        pos = {
            "x": -1 * self.ledge_position/2 * self.rolling_sliding_axis_length,
            "y": -0.5 * scale['y'],
            "z": 0
        }

        commands.extend(self.add_physics_object(
                    record=record,
                    position=pos,
                    rotation=TDWUtils.VECTOR3_ZERO,
                    mass=1000,
                    dynamic_friction=0.5,
                    static_friction=0.5,
                    bounciness=0.,
                    o_id=o_id,
                    add_data = True)
        )

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

    def _place_target_zone(self) -> List[dict]:
        if self.add_second_ramp:
            cmd = super()._place_target_zone()
            self.force_wait_zone = 0
            return cmd
        else:
            #self.zone_scale_range = {'x': 0.3, 'y':0.3, 'z':0.3}

            # create a target zone (usually flat, with same texture as room)

            record, data = self.random_primitive(self._zone_types,
                                                 scale={'x': 0.4, 'y':0.1, 'z':0.5},
                                                 color=self.zone_color,
                                                 add_data=False
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

            # place it just beyond the target object with an effectively immovable mass and high friction
            commands = []
            self.zone_location =  self._get_zone_location(scale)
            commands.extend(
                self.add_primitive(
                    record=record,
                    position=self.zone_location,
                    rotation=TDWUtils.VECTOR3_ZERO,
                    scale=scale,
                    material=self.zone_material,
                    color=rgb,
                    mass=0.5,
                    scale_mass=False,
                    dynamic_friction=self.zone_friction,
                    static_friction=(10.0 * self.zone_friction),
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

            self.force_wait_zone = 145
            self.hold_cmd = {"$type": "teleport_object",
                          "id": o_id,
                          "position": self.zone_location}

            return commands

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:


        if (self.force_wait_zone != 0) and frame <= self.force_wait_zone:
            if self.PRINT:
                print("applied %s at time step %d" % (self.hold_cmd, frame))
            output = [self.hold_cmd]
        else:
            output = []

        #print("frame", output)

        return output


    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > 400
    def _get_zone_location(self, scale):
        """Where to place the target zone? Right behind the target object."""
        # 0.1 for the probe shift, 0.01 to put under ramp
        if self.add_second_ramp:

            if self.zone_dloc == -1:
                rnd = random.uniform(0, 1)

                if rnd < 0.5:
                    return self._get_zone_second_location(scale)
            elif self.zone_dloc == 1:
                return self._get_zone_second_location(scale)

            elif self.zone_dloc == 2:
                print("zone dloc 2")

            else:
                raise ValueError

            return {
                "x": self.ramp_scale['z']*self.second_ramp_factor+self.zone_scale_range['x']/2,
                "y": 0.0 if not self.remove_zone else 10.0,
                "z":  random.uniform(-self.zjitter,self.zjitter) if not self.remove_zone else 10.0
            }
        else:

            if self.zone_dloc == 1:
                return {
                    "x": self.ramp_scale['z']*self.second_ramp_factor+self.zone_scale_range['x']/2 -random.uniform(2.3, 2.7), #3.1
                    "y": 2.8 if not self.remove_zone else 10.0,
                    "z": 0.45 + random.uniform(-self.zjitter,self.zjitter) if not self.remove_zone else 10.0
                }

            elif self.zone_dloc == 2:
                return {
                    "x": self.ramp_scale['z']*self.second_ramp_factor+self.zone_scale_range['x']/2 + 1.0,
                    "y": 2.8 if not self.remove_zone else 10.0,
                    "z": 0.25 + random.uniform(-self.zjitter,self.zjitter) if not self.remove_zone else 10.0
                }
            else:
                raise ValueError
    def _get_zone_second_location(self, scale):
        print(self.zone_scale)
        if self.add_second_ramp:
            SHIFT = self.ramp_scale['z']/100. + 0.02# 0.18
        else:
            SHIFT = self.ramp_scale['z']*0.07
        return {
            "x": SHIFT, #n_axis_length+ SHIFT,# + 0.5 * self.zone_scale_range['x'] + SHIFT,
            "y": self.zone_scale['x']/8. if not self.remove_zone else 10.0,
            "z":  random.uniform(-self.zjitter,self.zjitter) if not self.remove_zone else 10.0
        }

    def clear_static_data(self) -> None:
        Dominoes.clear_static_data(self)
        self.distinct_ids = np.empty(dtype=np.int32, shape=0)
        self.middle_type = None
        self.distractors = OrderedDict()
        self.occluders = OrderedDict()
        # clear some other stuff

    def _write_class_specific_data(self, static_group: h5py.Group) -> None:
        variables = static_group.create_group("variables")

        try:
            static_group.create_dataset("is_single_ramp", data=self.is_single_ramp)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("star_friction", data=self.zone_friction)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("star_id", data=self.star_id)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("cloth_size", data=self.cloth_size)
        except (AttributeError,TypeError):
            pass

        try:
            static_group.create_dataset("cloth_material", data=self.cloth_material)
        except (AttributeError,TypeError):
            pass



    def _write_static_data(self, static_group: h5py.Group) -> None:
        Dominoes._write_static_data(self, static_group)

        # static_group.create_dataset("bridge_height", data=self.bridge_height)

    @staticmethod
    def get_controller_label_funcs(classname = "Rolling_Sliding"):

        funcs = Dominoes.get_controller_label_funcs(classname)

        return funcs



if __name__ == "__main__":
    import platform, os

    args = get_rolling_sliding_args("rolling_sliding")

    print("gpu", args.gpu)
    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":" + str(args.gpu + 1)
        else:
            os.environ["DISPLAY"] = ":"
    ColC = FricRamp(
        room=args.room,
        randomize=args.random,
        seed=args.seed,
        target_zone=args.zone,
        zone_location=args.zlocation,
        zone_scale_range=args.zscale,
        zone_color=args.zcolor,
        zone_material=args.zmaterial,
        zone_friction=args.zfriction,
        zone_dloc = args.zdloc,
        platform_height = args.pheight,
        cloth_size = args.csize,
        target_objects=args.target,
        probe_objects=args.probe,
        target_scale_range=args.tscale,
        target_rotation_range=args.trot,
        probe_rotation_range=args.prot,
        probe_scale_range=args.pscale,
        probe_mass_range=args.pmass,
        target_color=args.color,
        probe_color=args.pcolor,
        collision_axis_length=args.collision_axis_length,
        force_scale_range=args.fscale,
        force_angle_range=args.frot,
        force_offset=args.foffset,
        force_offset_jitter=args.fjitter,
        force_wait=args.fwait,
        remove_target=bool(args.remove_target),
        remove_zone=bool(args.remove_zone),
        zjitter = args.zjitter,
        fupforce = args.fupforce,
        use_ramp = args.ramp,
        use_ledge = args.use_ledge,
        ledge = args.ledge,
        ledge_position = args.ledge_position,
        ledge_scale = args.ledge_scale,
        add_second_ramp = args.add_second_ramp,
        is_single_ramp = args.is_single_ramp,
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
        distractor_types=args.distractor,
        distractor_categories=args.distractor_categories,
        num_distractors=args.num_distractors,
        occluder_types=args.occluder,
        occluder_categories=args.occluder_categories,
        num_occluders=args.num_occluders,
        occlusion_scale=args.occlusion_scale,
        ramp_scale=args.ramp_scale,
        rolling_sliding_axis_length = args.rolling_sliding_axis_length,
        target_lift = args.tlift,
        flex_only=args.only_use_flex_objects,
        no_moving_distractors=args.no_moving_distractors,
        use_test_mode_colors=args.use_test_mode_colors
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
               save_meshes=args.save_meshes,
               write_passes=args.write_passes,
               args_dict=vars(args)
        )
    else:
        ColC.communicate({"$type": "terminate"})