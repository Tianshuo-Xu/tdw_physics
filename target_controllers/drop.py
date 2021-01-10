import h5py
import numpy as np
from enum import Enum
import random
from typing import List, Dict, Tuple
from weighted_collection import WeightedCollection
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelRecord
from tdw_physics.rigidbodies_dataset import RigidbodiesDataset
from tdw_physics.util import MODEL_LIBRARIES, get_args

class _StackType(Enum):
    """
    The stability type.
    """

    stable = 1
    maybe_stable = 2
    base_stable = 3
    unstable = 4


class Drop(RigidbodiesDataset):
    """
    Drop a random Flex primitive object on another random Flex primitive object
    """

    def __init__(self, port: int = 1071, height_range=[0.5, 1.5], drop_jitter=0.02, target_color=None, **kwargs):
        self._drop_types = MODEL_LIBRARIES["models_flex.json"].records
        self.height_range = height_range
        self.drop_jitter = drop_jitter
        self.target_color = target_color

        super().__init__(port=port, **kwargs)

        self.clear_static_data()

    def clear_static_data(self) -> None:
        super().clear_static_data()

        ## object colors and scales
        self.colors = np.empty(dtype=np.float32, shape=(0,3))
        self.scales = np.empty(dtype=np.float32, shape=0)
        self.heights = np.empty(dtype=np.float32, shape=0)
        self.target_type = self.drop_type = None

    def get_field_of_view(self) -> float:
        return 55

    def get_scene_initialization_commands(self) -> List[dict]:
        return [self.get_add_scene(scene_name="box_room_2018"),
                {"$type": "set_aperture",
                 "aperture": 4.8},
                {"$type": "set_post_exposure",
                 "post_exposure": 0.4},
                {"$type": "set_ambient_occlusion_intensity",
                 "intensity": 0.175},
                {"$type": "set_ambient_occlusion_thickness_modifier",
                 "thickness": 3.5}]

    def random_color(self):
        return [random.random(), random.random(), random.random()]

    def random_primitive(self, object_types: List[ModelRecord], scale: List[float] = [0.2, 0.3], color: List[float] = None) -> dict:
        obj_record = random.choice(object_types)
        obj_data = {
            "id": self.get_unique_id(),
            "scale": random.uniform(scale[0], scale[1]),
            "color": np.array(color or self.random_color()),
            "name": obj_record.name
        }
        self.scales = np.append(self.scales, obj_data["scale"])
        self.colors = np.concatenate([self.colors, obj_data["color"].reshape((1,3))], axis=0)
        return obj_record, obj_data

    def get_trial_initialization_commands(self) -> List[dict]:
        commands = []

        # Choose a stationary target object.
        target_obj = random.choice(self._drop_types)
        self.target_type = target_obj.name

        # Place it.
        commands.extend(self._place_target_object(target_obj))

        # Choose a drop object.
        drop_obj = random.choice(self._drop_types)
        self.drop_type = drop_obj.name

        # Drop it.
        commands.extend(self._drop_object(drop_obj))
        drop_height = drop_obj.bounds['top']['y']

        # Teleport the avatar to a reasonable position based on the height of the stack.
        # Look at the center of the stack, then jostle the camera rotation a bit.
        # Apply a slight force to the base object.
        a_pos = self.get_random_avatar_position(radius_min=drop_height,
                                                radius_max=1.3 * drop_height,
                                                y_min=drop_height / 4,
                                                y_max=drop_height / 3,
                                                center=TDWUtils.VECTOR3_ZERO)

        cam_aim = {"x": 0, "y": drop_height * 0.5, "z": 0}
        commands.extend([{"$type": "teleport_avatar_to",
                          "position": a_pos},
                         {"$type": "look_at_position",
                          "position": cam_aim},
                         {"$type": "set_focus_distance",
                          "focus_distance": TDWUtils.get_distance(a_pos, cam_aim)},
                         {"$type": "rotate_sensor_container_by",
                          "axis": "pitch",
                          "angle": random.uniform(-5, 5)},
                         {"$type": "rotate_sensor_container_by",
                          "axis": "yaw",
                          "angle": random.uniform(-5, 5)},
                         {"$type": "apply_force_to_object",
                          "force": {"x": random.uniform(-0.05, 0.05),
                                    "y": 0,
                                    "z": random.uniform(-0.05, 0.05)},
                          "id": int(self.object_ids[0])}])
        return commands

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:
        return []

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        ## color and scales of primitive objects
        static_group.create_dataset("color", data=self.colors)
        static_group.create_dataset("scale", data=self.scales)
        static_group.create_dataset("height", data=self.heights)
        static_group.create_dataset("target_type", data=self.target_type)
        static_group.create_dataset("drop_type", data=self.drop_type)

    def _write_frame(self, frames_grp: h5py.Group, resp: List[bytes], frame_num: int) -> \
            Tuple[h5py.Group, h5py.Group, dict, bool]:
        frame, objs, tr, sleeping = super()._write_frame(frames_grp=frames_grp, resp=resp, frame_num=frame_num)
        # If this is a stable structure, disregard whether anything is actually moving.
        return frame, objs, tr, sleeping and frame_num < 300

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > 300

    def _place_target_object(self, record: ModelRecord) -> List[dict]:
        """
        Place a primitive object at the room center.
        """

        # update data
        o_id = self.get_unique_id()
        scale = random.uniform(0.2, 0.3)
        rgb = np.array(self.random_color())

        self.scales = np.append(self.scales, scale)
        self.colors = np.concatenate([self.colors, rgb.reshape((1,3))], axis=0)

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


    # def _drop_object(self, record: ModelRecord, height: float, scale: float, color: List[float]) -> List[dict]:
    def _drop_object(self, record: ModelRecord) -> List[dict]:
        """
        Position a primitive object at some height and drop it.

        :param record: The object model record.
        :param height: The initial height from which to drop the object.
        :param scale: The scale of the object.

        :return: A list of commands to add the object to the simulation.
        """

        o_id = self.get_unique_id()

        # Set its properties
        scale = random.uniform(0.2, 0.3)
        rgb = np.array(self.random_color())
        height = random.uniform(self.height_range[0], self.height_range[1])

        # Add a record of the object scale, height, and color.
        self.heights = np.append(self.heights, height)
        self.scales = np.append(self.scales, scale)
        self.colors = np.concatenate([self.colors, rgb.reshape((1,3))], axis=0)

        # Add the object with random physics values.
        commands = []
        commands.extend(
            self.add_physics_object(
                record=record,
                position={
                    "x": random.uniform(-self.drop_jitter, self.drop_jitter),
                    "y": height,
                    "z": random.uniform(-self.drop_jitter, self.drop_jitter)
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

if __name__ == "__main__":
    MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
    print("models", MODEL_NAMES)
    from argparse import ArgumentParser
    # args = get_args("drop")
    common_parser = get_args("drop", return_parser=True)
    parser = ArgumentParser(parents=[common_parser])
    parser.add_argument("--drop", type=str, default=MODEL_NAMES[0], help="comma-separated list of possible drop objects")
    args = parser.parse_args()

    print("drop object", args.drop)

    DC = Drop(randomize=args.random, seed=args.seed, drop_jitter=0.1)
    if bool(args.run):
        DC.run(num=args.num, output_dir=args.dir, temp_path=args.temp, width=args.width, height=args.height)
    else:
        DC.communicate({"$type": "terminate"})
