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
    ### zone
    parser.add_argument("--zscale",
                        type=str,
                        default="1.0,0.01,1.0",
                        help="scale of target zone")

    parser.add_argument("--zone",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible target zone shapes")

    parser.add_argument("--zjitter",
                        type=float,
                        default=0.35,
                        help="amount of z jitter applied to the target zone")

    ### probe
    parser.add_argument("--probe",
                        type=str,
                        default="sphere",
                        help="comma-separated list of possible probe objects")

    parser.add_argument("--pscale",
                        type=str,
                        default="0.35,0.35,0.35",
                        help="scale of probe objects")

    parser.add_argument("--plift",
                        type=float,
                        default=0.,
                        help="Lift the probe object off the floor. Useful for rotated objects")

    ### force
    parser.add_argument("--fscale",
                        type=str,
                        default="[5.0,5.0]",
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
                        default=0.1,
                        help="jitter around object centroid to apply force")

    ###target
    parser.add_argument("--target",
                        type=str,
                        default="pipe,cube,pentagon",
                        help="comma-separated list of possible target objects")

    parser.add_argument("--tscale",
                        type=str,
                        default="0.3,0.5,0.3",
                        help="scale of target objects")

    ### layout
    parser.add_argument("--collision_axis_length",
                        type=float,
                        default=2.0,
                        help="Length of spacing between probe and target objects at initialization.")

    ## collision specific arguments
    parser.add_argument("--fupforce",
                        type=str,
                        default='[0,0]',
                        help="Upwards component of force applied, with 0 being purely horizontal force and 1 being the same force being applied horizontally applied vertically.")

    ## camera
    parser.add_argument("--camera_min_angle",
                        type=float,
                        default=45,
                        help="minimum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_max_angle",
                        type=float,
                        default=225,
                        help="maximum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_distance",
                        type=none_or_str,
                        default="1.75",
                        help="radial distance from camera to centerpoint")

    ## occluders and distractors
    parser.add_argument("--occluder_aspect_ratio",
                        type=none_or_str,
                        default="[0.5,2.5]",
                        help="The range of valid occluder aspect ratios")
    parser.add_argument("--distractor_aspect_ratio",
                        type=none_or_str,
                        default="[0.25,5.0]",
                        help="The range of valid distractor aspect ratios")
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
        return args

    args = parser.parse_args()
    args = domino_postproc(args)
    args = postprocess(args)

    return args

class WaterPush(Dominoes):

    def __init__(self,
                 port: int = None,
                 zjitter = 0,
                 fupforce = [0.,0.],
                 probe_lift = 0.,
                 **kwargs):
        # initialize everything in common w / Multidominoes
        super().__init__(port=port, **kwargs)
        self.zjitter = zjitter
        self.fupforce = fupforce
        self.probe_lift = probe_lift
        self.use_obi = True


        self._star_types = self._target_types
        self.star_scale_range = self.target_scale_range
        self._candidate_types = self._probe_types
        self.candidate_scale_range = self.probe_scale_range

        self.force_wait_range = [3, 3]
        self.obi_unique_ids = 0

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
        self.star_object["mass"] = 0.5 #2 * 10 ** np.random.uniform(-1,1) #random.choice([0.1, 2.0, 10.0])
        self.star_object["scale"] = get_random_xyz_transform(self.star_scale_range)
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

        # Choose and place a target object.
        commands.extend(self._place_star_object(interact_id))

        # Set the probe color
        if self.probe_color is None:
            self.probe_color = self.target_color if (self.monochrome and self.match_probe_and_target_color) else None

        # Choose, place, and push a probe object.
        commands.extend(self._place_and_push_probe_object(interact_id))

        # Build the intermediate structure that captures some aspect of "intuitive physics."
        commands.extend(self._build_intermediate_structure(interact_id))

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


    def _build_intermediate_structure(self, interact_id) -> List[dict]:

        # print("middle color", self.middle_color)
        # if self.randomize_colors_across_trials:
        #     self.middle_color = self.random_color(exclude=self.target_color) if self.monochrome else None

        commands = []

        # Go nuts
        # commands.extend(self._place_barrier_foundation())
        # commands.extend(self._build_bridge())

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
        star_position = {
            "x": self.collision_axis_length - 1.0,
            "y": 0. if not self.remove_target else 10.0,
            "z": 0. if not self.remove_target else 10.0
        }
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
                dynamic_friction=0.5,
                static_friction=0.5,
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

        return commands


    def _place_and_push_probe_object(self, interact_id) -> List[dict]:
        """
        Place a probe object at the other end of the collision axis, then apply a force to push it.
        """
        from tdw.obi_data.fluids.fluid import Fluid, FLUIDS
        from tdw.add_ons.third_person_camera import ThirdPersonCamera
        from tdw.obi_data.fluids.disk_emitter import DiskEmitter


        distinct_id = 0
        probe_type = self.candidate_dict[distinct_id]["type"]
        probe_scale = self.candidate_dict[distinct_id]["scale"]
        probe_mass = self.candidate_dict[distinct_id]["mass"]
        probe_color = self.candidate_dict[distinct_id]["color"]
        o_id = self._get_next_object_id() + self.obi_unique_ids * 5
        self.obi_unique_ids += 1

        vis = [1.5, 0.00001, 0.001, 0.01, 1.0]
        fluid = Fluid(
        capacity=1500,
        resolution=1.0,
        color={'a': 0.5, 'b': 0.995, 'g': 0.2, 'r': 0.2},
        rest_density=1.0,
        radius_scale=1.6, #2.0
        random_velocity=vis[3],
        smoothing=vis[0], #3.5, #3.0 #2.0 is like water, higher means stickier
        surface_tension=1.0,
        viscosity= vis[1], #0.001, #1.5
        vorticity=vis[4], #0.7
        reflection=0.25,
        transparency=0.2,
        refraction=-0.034,
        buoyancy=-1,
        diffusion=0,
        diffusion_data={'w': 0, 'x': 0, 'y': 0, 'z': 0},
        atmospheric_drag=0,
        atmospheric_pressure=0,
        particle_z_write=False,
        thickness_cutoff=vis[2],
        thickness_downsample=2,
        blur_radius=0.02,
        surface_downsample=1,
        render_smoothness=0.8,
        metalness=0,
        ambient_multiplier=1,
        absorption=5,
        refraction_downsample=1,
        foam_downsample=1,
        )

        fluid_start_position = {
            "x": self.collision_axis_length - 3.0,
            "y": 1.4,
            "z": 0
        }
        self.obi.create_fluid(object_id = o_id,
                 fluid=fluid,
                 shape=DiskEmitter(),
                 position=fluid_start_position, # y is height,
                 rotation={"x": 160, "y": -90, "z": 0}, #150, -90, 0
                 lifespan=1000,
                 speed=10)

        commands = []
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

    ColC = WaterPush(
        port=args.port,
        room=args.room,
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
        occluder_aspect_ratio=args.occluder_aspect_ratio,
        distractor_aspect_ratio=args.distractor_aspect_ratio,
        probe_lift = args.plift,
        flex_only=args.only_use_flex_objects,
        no_moving_distractors=args.no_moving_distractors,
        match_probe_and_target_color=args.match_probe_and_target_color,
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
