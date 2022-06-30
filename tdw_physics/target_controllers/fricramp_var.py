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
from tdw.output_data import OutputData, Transforms
from tdw_physics.rigidbodies_dataset import (RigidbodiesDataset,
                                             get_random_xyz_transform,
                                             get_range,
                                             handle_random_transform_args)
from tdw_physics.util import MODEL_LIBRARIES, get_parser, xyz_to_arr, arr_to_xyz, str_to_xyz

from tdw_physics.target_controllers.dominoes_var import Dominoes, MultiDominoes, get_args, none_or_str, none_or_int
from tdw_physics.postprocessing.labels import is_trial_valid

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}

OCCLUDER_CATS = "coffee table,houseplant,vase,chair,dog,sofa,flowerpot,coffee maker,stool,laptop,laptop computer,globe,bookshelf,desktop computer,garden plant,garden plant,garden plant"
DISTRACTOR_CATS = "coffee table,houseplant,vase,chair,dog,sofa,flowerpot,coffee maker,stool,laptop,laptop computer,globe,bookshelf,desktop computer,garden plant,garden plant,garden plant"

def get_collision_args(dataset_dir: str, parse=True):

    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')

    ## Changed defaults
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

class FricRamp(Dominoes):

    def __init__(self,
                 port: int = None,
                 zjitter = 0,
                 fupforce = [0.,0.],
                 probe_lift = 0.,
                 ramp_scale = [0.2,0.25,0.5],
                 rolling_sliding_axis_length = 1.15,
                 use_ramp = True,
                 use_ledge = False,
                 ledge = ['cube'],
                 ledge_position = 0.5,
                 ledge_scale = [100,0.1,0.1],
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
        self.DEFAULT_RAMPS = [MODEL_LIBRARIES["models_flex.json"].get_record("triangular_prism")]

        self.add_second_ramp = False
        self.second_ramp_factor = 0.5
        self.target_lift = target_lift
        """
        The color of the second ramp
        """
        self.second_ramp_color = [1., 1., 1.] #White
        self.fric_range=[0.6, 0.6]
        self.ramp_y_range=[0.7,0.95]

        self.probe_lift = probe_lift
        self.use_obi = False


        self._star_types = self._target_types
        self.star_scale_range = self.target_scale_range
        self._candidate_types = self._probe_types
        self.candidate_scale_range = self.probe_scale_range

        self.force_wait_range = [3, 3]

        self.is_single_ramp = False
        self.add_second_ramp = True

    def get_trial_initialization_commands(self, interact_id) -> List[dict]:
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

        self._sample_ramp_param()

        # Choose and place a target object.
        commands.extend(self._place_star_object(interact_id))

        commands.extend(self._place_ramp(interact_id))

        # Teleport the avatar to a reasonable position
        if interact_id == 0:
            self.a_pos = self.get_random_avatar_position(radius_min=self.camera_radius_range[0],
                                                radius_max=self.camera_radius_range[1],
                                                angle_min=self.camera_min_angle,
                                                angle_max=self.camera_max_angle,
                                                y_min=self.camera_min_height,
                                                y_max=self.camera_max_height,
                                                center=TDWUtils.VECTOR3_ZERO)

        # Set the camera parameters


        self._set_avatar_attributes(self.a_pos)

        commands.extend([
            {"$type": "teleport_avatar_to",
             "position": self.a_pos},
            {"$type": "look_at_position",
             "position": self.camera_aim},
            {"$type": "set_focus_distance",
             "focus_distance": TDWUtils.get_distance(self.a_pos, self.camera_aim)}
        ])

        # self.camera_position = a_pos
        # self.camera_rotation = np.degrees(np.arctan2(a_pos['z'], a_pos['x']))
        # dist = TDWUtils.get_distance(a_pos, self.camera_aim)
        # self.camera_altitude = np.degrees(np.arcsin((a_pos['y'] - self.camera_aim['y'])/dist))

        # Place distractor objects in the background
        commands.extend(self._place_background_distractors())

        # Place occluder objects in the background
        commands.extend(self._place_occluders())

        # test mode colors
        if self.use_test_mode_colors:
            self._set_test_mode_colors(commands)

        return commands

    def _sample_ramp_param(self):

        self.zone_friction = random.uniform(*self.fric_range)
        self.ramp_scale['y'] = random.uniform(*self.ramp_y_range)

        self.ramp = random.choice(self.DEFAULT_RAMPS)
        star_position = {"x": -self.rolling_sliding_axis_length, "y": self.target_lift, "z": 0.}
        ramp_pos = copy.deepcopy(star_position)
        ramp_pos['y'] = self.zone_scale['y'] if not self.remove_zone else 0.0 # don't intersect w zone

        self.ramp_pos = ramp_pos
        # figure out scale
        r_len, r_height, r_dep = self.get_record_dimensions(self.ramp)

        scale_x = (0.75 * self.collision_axis_length) / r_len
        if self.ramp_scale is None:
            self.ramp_scale = arr_to_xyz([scale_x, self.scale_to(r_height, 1.5), 0.75 * scale_x])
        self.ramp_end_x = self.ramp_pos['x'] + self.ramp_scale['x'] * r_len * 0.5
        # optionally add base

        if self.is_single_ramp:
            self.ramp_base_height = 0
        else:
            self.ramp_base_height_range = self.ramp_scale['y']
            self.ramp_base_height = random.uniform(*get_range(self.ramp_base_height_range))


    def _place_ramp_under_probe(self) -> List[dict]:

        cmds = []

        if self.is_single_ramp:

            r_len, r_height, r_dep = self.get_record_dimensions(self.ramp)
            second_ramp_pos = copy.deepcopy(self.ramp_pos)
            second_ramp_pos['y'] = self.zone_scale['y']*1.1 if not self.remove_zone else 0.0 # don't intersect w zone
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

            second_ramp_pos['x']  -= 1.0 #  - second_ramp_scale['z']

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



        else:
            # ramp params

            rgb = self.ramp_color or self.random_color(exclude=self.target_color)

            ramp_rot = self.get_y_rotation([90,90])
            ramp_id = self._get_next_object_id()

            self.ramp_rot = ramp_rot
            self.ramp_id = ramp_id
            r_len, r_height, r_dep = self.get_record_dimensions(self.ramp)

            # figure out scale
            # r_len, r_height, r_dep = self.get_record_dimensions(self.ramp)

            # scale_x = (0.75 * self.collision_axis_length) / r_len
            # if self.ramp_scale is None:
            #     self.ramp_scale = arr_to_xyz([scale_x, self.scale_to(r_height, 1.5), 0.75 * scale_x])
            # self.ramp_end_x = self.ramp_pos['x'] + self.ramp_scale['x'] * r_len * 0.5

            # # optionally add base
            # self.ramp_base_height_range = self.ramp_scale['y']
            cmds.extend(self._add_ramp_base_to_ramp(color=self.second_ramp_color, sample_ramp_base_height=False))
            # self.ramp_base_height = random.uniform(*get_range(self.ramp_base_height_range))

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


            # self.probe_initial_position['x'] += self.ramp_scale['z']*0.1
            # self.probe_initial_position['y'] = self.ramp_scale['y'] * r_height + self.ramp_base_height + self.probe_initial_position['y']



            if self.add_second_ramp:
                second_ramp_pos = copy.deepcopy(self.ramp_pos)
                second_ramp_pos['y'] = self.zone_scale['y']*1.1 if not self.remove_zone else 0.0 # don't intersect w zone


                second_ramp_id = self._get_next_object_id()
                scale_x = (0.75 * self.collision_axis_length) / r_len
                if self.ramp_scale is None:
                    self.ramp_scale = arr_to_xyz([scale_x, self.scale_to(r_height, 1.5), 0.75 * scale_x])
                second_ramp_scale = copy.deepcopy(self.ramp_scale)
                second_ramp_scale['x'] = self.ramp_scale['x']
                second_ramp_scale['y'] = self.second_ramp_factor*self.ramp_scale['y']
                second_ramp_scale['z'] = self.second_ramp_factor*self.ramp_scale['z']

                second_ramp_pos['x']  = second_ramp_scale['z']/2.

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


        return cmds

    def _build_intermediate_structure(self, interact_id) -> List[dict]:

        # print("middle color", self.middle_color)
        # if self.randomize_colors_across_trials:
        #     self.middle_color = self.random_color(exclude=self.target_color) if self.monochrome else None

        commands = []

        # Go nuts
        # commands.extend(self._place_barrier_foundation())
        # commands.extend(self._build_bridge())

        return commands

    def _place_ramp(self, interact_id):
        commands = []
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
        return commands


    def _place_star_object(self, interact_id) -> List[dict]:
        """
        Place a primitive object at one end of the collision axis.
        """



        distinct_id = 1
        self.distinct_ids = np.append(self.distinct_ids, distinct_id)
        # create a target object
        #if not self.repeat_trial: # sample from scratch
        star_type = self.star_object["type"]
        star_scale = self.star_object["scale"]
        star_mass = self.star_object["mass"]
        star_color = self.star_object["color"]

        # select an object
        # record, data = self.random_primitive(self._target_types,
        #                                      scale=self.target_scale_range,
        #                                      color=self.target_color,
        #                                      add_data=False
        # )

        record, data = self.random_primitive([star_type],
                                             scale=star_scale,
                                             color=star_color,
                                             add_data=False
        )
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

        assert(o_id == 2), "make sure the star object is always with the same id"
        self.star_id = o_id

        # else:
        #     # object properties don't change
        #     record = self.target
        #     scale = self.target_scale
        #     o_id = self.target_id
        #     rgb = self.target_color


        if any((s <= 0 for s in scale.values())):
            self.remove_target = True

        # Where to put the target
        pos_id = 0
        # carpet, target, middle, middle | prob
        star_position = {"x": -self.rolling_sliding_axis_length, "y": self.target_lift, "z": 0.}

        r_len, r_height, r_dep = self.get_record_dimensions(self.ramp)
        star_position['x'] += self.ramp_scale['z']*0.1
        star_position['y'] += self.ramp_scale['y'] * r_height + self.ramp_base_height

        star_rotation = self.get_rotation(self.target_rotation_range)

        self.star_pos_id = pos_id

        self.middle_scale = scale

        self.target = record
        self.target_type = data["name"]
        self.target_color = rgb
        self.target_scale = scale
        self.target_id = o_id
        if self.target_rotation is None:
            self.target_rotation = star_rotation
        if self.target_position is None:
            self.target_position = star_position

        # Commands for adding hte object
        commands = []

        commands.extend(
            self.add_primitive(
                record=record,
                position=star_position,
                rotation=star_rotation,
                scale=scale,
                material=self.target_material,
                color=rgb,
                mass=star_mass, #2.0,
                scale_mass=False,
                dynamic_friction=0.1,
                static_friction=0.1,
                bounciness=0.0,
                o_id=o_id,
                add_data=(not self.remove_target),
                make_kinematic=False
            ))

        # If this scene won't have a target
        if self.remove_target:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]


        commands.extend([
            # {"$type": "set_object_collision_detection_mode",
            #  "mode": "continuous_speculative",
            #  "id": o_id},
            {"$type": "set_object_drag",
             "id": o_id,
             "drag": 0, "angular_drag": 0}])


        # Apply a force to the target object
        self.push_force = self.get_push_force(
            scale_range=star_mass * np.array(self.force_scale_range),
            angle_range=self.force_angle_range,
            yforce=self.fupforce)
        self.push_force = self.rotate_vector_parallel_to_floor(
            self.push_force, -star_rotation['y'], degrees=True)

        self.push_position = star_position
        if self.use_ramp:
            self.push_cmd = {
                "$type": "apply_force_to_object",
                "force": self.push_force,
                "id": int(o_id)
            }
        else:
            self.push_position = {
                k:v+self.force_offset[k]*self.rotate_vector_parallel_to_floor(
                    self.target_scale, star_rotation['y'])[k]
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
        self.force_wait = int(random.uniform(*get_range(self.force_wait_range)))
        print("force wait", self.force_wait)

        # if self.force_wait == 0:
        #     commands.append(self.push_cmd)

        return commands



    def _get_zone_location(self, scale):
        """Where to place the target zone? Right behind the target object."""
        BUFFER = 0
        return {
            "x": self.collision_axis_length,# + 0.5 * self.zone_scale_range['x'] + BUFFER,
            "y": 0.0 if not self.remove_zone else 10.0,
            "z":  random.uniform(-self.zjitter,self.zjitter) if not self.remove_zone else 10.0
        }



    def clear_static_data(self) -> None:
        super().clear_static_data()

        self.distinct_ids = np.empty(dtype=np.int32, shape=0)
        self.distractors = OrderedDict()
        self.occluders = OrderedDict()
        # clear some other stuff

    def _write_static_data(self, static_group: h5py.Group) -> None:
        Dominoes._write_static_data(self, static_group)

    @staticmethod
    def get_controller_label_funcs(classname = "Collision"):

        funcs = Dominoes.get_controller_label_funcs(classname)

        return funcs

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > 200 # End after X frames even if objects are still moving.

    def _set_distractor_attributes(self) -> None:

        self.distractor_angular_spacing = 20
        self.distractor_distance_fraction = [0.4,1.0]
        self.distractor_rotation_jitter = 30
        self.distractor_min_z = self.middle_scale['z'] * 2.0
        self.distractor_min_size = 0.5
        self.distractor_max_size = 1.0

    def _set_occlusion_attributes(self) -> None:

        self.occluder_angular_spacing = 15
        self.occlusion_distance_fraction = [0.6, 0.8]
        self.occluder_rotation_jitter = 30.
        self.occluder_min_z = self.middle_scale['z'] * 2.0
        self.occluder_min_size = 0.25
        self.occluder_max_size = 1.0
        self.rescale_occluder_height = True


if __name__ == "__main__":
    import platform, os

    args = get_collision_args("collision")

    # if platform.system() == 'Linux':
    #     if args.gpu is not None:
    #         os.environ["DISPLAY"] = ":0." + str(args.gpu)
    #     else:
    #         os.environ["DISPLAY"] = ":0"

    ColC = FricRamp(
        port=args.port,
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
                 framerate=args.framerate,
                 save_passes=args.save_passes.split(','),
                 save_movies=args.save_movies,
                 save_labels=args.save_labels,
                 save_meshes=args.save_meshes,
                 write_passes=args.write_passes,
                 args_dict=vars(args)
        )
    else:
        ColC.communicate({"$type": "terminate"})
