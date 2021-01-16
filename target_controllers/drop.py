from argparse import ArgumentParser
import h5py
import numpy as np
from enum import Enum
import random
from typing import List, Dict, Tuple
from weighted_collection import WeightedCollection
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelRecord
from tdw_physics.rigidbodies_dataset import RigidbodiesDataset
from tdw_physics.util import MODEL_LIBRARIES, get_parser, xyz_to_arr, arr_to_xyz


MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]

def get_args(dataset_dir: str):
    """
    Combine Drop-specific arguments with controller-common arguments
    """
    common = get_parser(dataset_dir, get_help=False)
    parser = ArgumentParser(parents=[common])

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
                        default=0.75,
                        help="min height to drop object from")
    parser.add_argument("--ymax",
                        type=float,
                        default=1.25,
                        help="max height to drop object from")
    parser.add_argument("--smin",
                        type=float,
                        default=0.2,
                        help="min scale of target and drop objects")
    parser.add_argument("--smax",
                        type=float,
                        default=0.3,
                        help="max scale of target and drop objects")
    parser.add_argument("--rot",
                        type=str,
                        default=None,
                        help="comma separated list of initial drop rotation values")
    parser.add_argument("--jitter",
                        type=float,
                        default=0.2,
                        help="amount to jitter initial drop object horizontal position across trials")
    parser.add_argument("--color",
                        type=str,
                        default=None,
                        help="comma-separated R,G,B values for the target object color. Defaults to random.")
    parser.add_argument("--camera_distance",
                        type=float,
                        default=1.25,
                        help="radial distance from camera to drop/target object pair")
    parser.add_argument("--camera_min_height",
                        type=float,
                        default=1./3,
                         help="min height of camera as a fraction of drop height")
    parser.add_argument("--camera_max_height",
                        type=float,
                        default=2./3,
                        help="max height of camera as a fraction of drop height")

    args = parser.parse_args()

    # whether to set all objects same color
    args.monochrome = bool(args.monochrome)

    if args.rot is not None:
        rot = list(map(float, args.rot.split(',')))
        assert len(rot) == 3, rot
        args.rot = rot

    if args.drop is not None:
        drop_list = args.drop.split(',')
        assert all([d in MODEL_NAMES for d in drop_list]), \
            "All drop object names must be elements of %s" % MODEL_NAMES
        args.drop = drop_list
    else:
        args.drop = MODEL_NAMES

    if args.target is not None:
        targ_list = args.target.split(',')
        assert all([t in MODEL_NAMES for t in targ_list]), \
            "All target object names must be elements of %s" % MODEL_NAMES
        args.target = targ_list
    else:
        args.target = MODEL_NAMES

    if args.color is not None:
        rgb = [float(c) for c in args.color.split(',')]
        assert len(rgb) == 3, rgb
        args.color = rgb

    return args

class Drop(RigidbodiesDataset):
    """
    Drop a random Flex primitive object on another random Flex primitive object
    """

    def __init__(self,
                 port: int = 1071,
                 drop_objects=MODEL_NAMES,
                 target_objects=MODEL_NAMES,
                 height_range=[0.5, 1.5],
                 scale_range=[0.2, 0.3],
                 drop_jitter=0.02,
                 drop_rotation=None,
                 target_color=None,
                 camera_radius=1.0,
                 camera_min_height=1./3,
                 camera_max_height=2./3,
                 **kwargs):

        ## allowable object types
        self._drop_types = [r for r in MODEL_LIBRARIES["models_flex.json"].records if r.name in drop_objects]
        self._target_types = [r for r in MODEL_LIBRARIES["models_flex.json"].records if r.name in target_objects]

        ## object properties
        self.height_range = height_range
        self.scale_range = scale_range
        self.drop_jitter = drop_jitter
        self.target_color = target_color
        self.drop_rotation = drop_rotation
        
        ## camera properties
        self.camera_radius = camera_radius
        self.camera_min_height = camera_min_height
        self.camera_max_height = camera_max_height

        ## initializes static data and RNG
        super().__init__(port=port, **kwargs)

    def clear_static_data(self) -> None:
        super().clear_static_data()

        ## scenario-specific metadata: object types and drop position
        self.heights = np.empty(dtype=np.float32, shape=0)
        self.target_type = None
        self.drop_type = None
        self.drop_position = None
        self.drop_rotation = None

    def get_field_of_view(self) -> float:
        return 55

    def get_scene_initialization_commands(self) -> List[dict]:
        return [self.get_add_scene(scene_name="box_room_2018"),
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

        # Choose and place a target object.
        commands.extend(self._place_target_object())

        # Choose and drop an object.
        commands.extend(self._drop_object())

        # Teleport the avatar to a reasonable position based on the drop height.
        a_pos = self.get_random_avatar_position(radius_min=self.camera_radius,
                                                radius_max=self.camera_radius,
                                                y_min=self.drop_height * self.camera_min_height,
                                                y_max=self.drop_height * self.camera_max_height,
                                                center=TDWUtils.VECTOR3_ZERO)

        cam_aim = {"x": 0, "y": self.drop_height * 0.5, "z": 0}
        commands.extend([
            {"$type": "teleport_avatar_to",
             "position": a_pos},
            {"$type": "look_at_position",
             "position": cam_aim},
            {"$type": "set_focus_distance",
             "focus_distance": TDWUtils.get_distance(a_pos, cam_aim)}
        ])
        return commands

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:
        return []

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        ## color and scales of primitive objects
        static_group.create_dataset("target_type", data=self.target_type)
        static_group.create_dataset("drop_type", data=self.drop_type)
        static_group.create_dataset("drop_position", data=xyz_to_arr(self.drop_position))
        static_group.create_dataset("drop_rotation", data=xyz_to_arr(self.drop_rotation))

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
        return frame > 300

    def _place_target_object(self) -> List[dict]:
        """
        Place a primitive object at the room center.
        """

        # create a target object
        record, data = self.random_primitive(self._target_types,
                                             scale=self.scale_range,
                                             color=self.target_color)
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        self.target_type = data["name"]
        self.object_color = rgb if self.monochrome else None

        # add the object
        commands = []
        commands.extend(
            self.add_physics_object(
                record=record,
                position={
                    "x": 0.,
                    "y": 0.,
                    "z": 0.
                },
                rotation={
                    "x": 0,
                    "y": random.uniform(0, 360),
                    "z": 0
                },
                mass=random.uniform(2,7),
                dynamic_friction=random.uniform(0, 0.9),
                static_friction=random.uniform(0, 0.9),
                bounciness=random.uniform(0, 1),
                o_id=o_id))


        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": o_id},
            {"$type": "scale_object",
             "scale_factor": {p: scale for p in ["x", "y", "z"]},
             "id": o_id}])

        return commands

    def _drop_object(self) -> List[dict]:
        """
        Position a primitive object at some height and drop it.

        :param record: The object model record.
        :param height: The initial height from which to drop the object.
        :param scale: The scale of the object.


        :return: A list of commands to add the object to the simulation.
        """

        # Create an object to drop.
        record, data = self.random_primitive(self._drop_types,
                                             scale=self.scale_range,
                                             color=self.object_color)
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        self.drop_type = data["name"]

        # Choose the drop position and pose.
        height = random.uniform(self.height_range[0], self.height_range[1])
        self.heights = np.append(self.heights, height)
        self.drop_height = height
        self.drop_position = {
            "x": random.uniform(-self.drop_jitter, self.drop_jitter),
            "y": height,
            "z": random.uniform(-self.drop_jitter, self.drop_jitter)
        }
        #XXX TODO: more flexible API for specifying initial rotation
        if self.drop_rotation is None:
            self.drop_rotation = {"x": 0,  #should this also be random??
                                  "y": random.uniform(0,360),
                                  "z": 0   #also random??
            }

        # Add the object with random physics values.
        commands = []
        commands.extend(
            self.add_physics_object(
                record=record,
                position=self.drop_position,
                rotation=self.drop_rotation,
                mass=random.uniform(2,7),
                dynamic_friction=random.uniform(0, 0.9),
                static_friction=random.uniform(0, 0.9),
                bounciness=random.uniform(0, 1),
                o_id=o_id))

        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": o_id},
            {"$type": "scale_object",
             "scale_factor": {p: scale for p in ["x", "y", "z"]},
             "id": o_id}])

        return commands

if __name__ == "__main__":

    args = get_args("drop")
    print("all object types", MODEL_NAMES)
    print("drop objects", args.drop)
    print("target objects", args.target)

    DC = Drop(
        randomize=args.random,
        seed=args.seed,
        height_range=[args.ymin, args.ymax],
        scale_range=[args.smin, args.smax],
        drop_jitter=args.jitter,
        drop_rotation=args.rot,
        drop_objects=args.drop,
        target_objects=args.target,
        target_color=args.color,
        camera_radius=args.camera_distance,
        camera_min_height=args.camera_min_height,
        camera_max_height=args.camera_max_height,
        monochrome=args.monochrome
    )
    if bool(args.run):
        DC.run(num=args.num,
               output_dir=args.dir,
               temp_path=args.temp,
               width=args.width,
               height=args.height)
    else:
        DC.communicate({"$type": "terminate"})
