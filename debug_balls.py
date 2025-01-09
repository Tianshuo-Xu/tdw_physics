from pathlib import Path
import random
import numpy as np
from typing import List, Dict
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from tdw_physics.rigidbodies_dataset import RigidbodiesDataset
from tdw_physics.object_position import ObjectPosition
from tdw_physics.util import get_args


class BallsDataset(RigidbodiesDataset):
    """
    Per trial, create 4 or 5 balls. 
    Apply a force to one of them, directed at another.
    Per frame, save object/physics metadata and image data.
    """

    def __init__(self, port: int = 1071):
        # Load a library of primitive shapes that includes a sphere.
        # "prim.json" is a default library in TDW that includes a "prim_sphere" record.
        lib = ModelLibrarian(str(Path("prim.json").resolve()))
        # Filter out just the sphere records, in case there are multiple.
        # For example, the standard "prim_sphere" or "prim_sphere_smooth".
        self.sphere_records = [r for r in lib.records if "sphere" in r.name]
        # We only need one, but you could keep multiple references if you want random sphere types.
        self.record = self.sphere_records[0]

        self._target_id: int = 0

        super().__init__(port=port)

    def get_field_of_view(self) -> float:
        return 55

    def get_scene_initialization_commands(self) -> List[dict]:
        """
        Scene-wide setup commands (runs once at the start).
        """
        return [
            self.get_add_scene(scene_name="box_room_2018"),
            {"$type": "set_aperture", "aperture": 4.8},
            {"$type": "set_focus_distance", "focus_distance": 1.25},
            {"$type": "set_post_exposure", "post_exposure": 0.4},
            {"$type": "set_ambient_occlusion_intensity", "intensity": 0.175},
            {"$type": "set_ambient_occlusion_thickness_modifier", "thickness": 3.5}
        ]

    def get_trial_initialization_commands(self) -> List[dict]:
        """
        Called at the start of each trial. 
        Spawns 4 or 5 balls, applies force from ball 0 to ball 1, 
        and positions the avatar (camera).
        """
        num_objects = random.choice([4, 5])
        object_positions: List[ObjectPosition] = []
        commands = []

        # Add 4 or 5 sphere objects.
        for i in range(num_objects):
            o_id = Controller.get_unique_id()

            # Random scale factor. For a sphere in prim.json, 
            # you can just do uniform scales between 0.4 and 1.0, for example.
            scale = random.uniform(0.4, 1.0)

            # Get a random position that doesn't collide with existing objects.
            o_pos = self._get_object_position(object_positions=object_positions)
            object_positions.append(ObjectPosition(position=o_pos, radius=scale))

            # Add the physics sphere object with random friction, bounciness, etc.
            commands.extend(
                self.add_physics_object(
                    o_id=o_id,
                    record=self.record,
                    position=o_pos,
                    rotation={"x": 0, "y": random.uniform(-180, 180), "z": 0},
                    mass=random.uniform(1, 5),
                    dynamic_friction=random.uniform(0, 0.9),
                    static_friction=random.uniform(0, 0.9),
                    bounciness=random.uniform(0, 1)
                )
            )

            # Scale the object.
            commands.append({
                "$type": "scale_object",
                "id": o_id,
                "scale_factor": {"x": scale, "y": scale, "z": scale}
            })

        # Now choose which ball will apply the force, and which will be the target.
        force_id = int(self.object_ids[0])
        self._target_id = int(self.object_ids[1])

        # Aim ball 0 at ball 1, apply force, then adjust the avatar.
        commands.extend([
            # Orient "force ball" so it looks at the target ball.
            {"$type": "object_look_at",
             "other_object_id": self._target_id,
             "id": force_id},
            {"$type": "rotate_object_by",
             "angle": random.uniform(-5, 5),
             "id": force_id,
             "axis": "yaw",
             "is_world": True},
            {"$type": "apply_force_magnitude_to_object",
             "magnitude": random.uniform(20, 60),
             "id": force_id},

            # Move the avatar to a random vantage point.
            {"$type": "teleport_avatar_to",
             "position": self.get_random_avatar_position(
                 radius_min=0.9, radius_max=1.5, y_min=0.5, y_max=1.25,
                 center=TDWUtils.VECTOR3_ZERO
             )},
            {"$type": "look_at",
             "object_id": self._target_id,
             "use_centroid": True},

            # Slight random pitch or yaw to the camera.
            {"$type": "rotate_sensor_container_by",
             "axis": "pitch",
             "angle": random.uniform(-5, 5)},
            {"$type": "rotate_sensor_container_by",
             "axis": "yaw",
             "angle": random.uniform(-5, 5)},

            # Focus camera on the target ball.
            {"$type": "focus_on_object", "object_id": self._target_id}
        ])

        return commands

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:
        """
        Optional per-frame commands. We keep the camera 
        focused on the target ball every frame.
        """
        return [
            {"$type": "focus_on_object",
             "object_id": self._target_id}
        ]

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        """
        End the trial after 1000 frames (for example).
        """
        return frame > 1000

    @staticmethod
    def _get_object_position(object_positions: List[ObjectPosition],
                             max_tries: int = 1000, radius: float = 3) -> Dict[str, float]:
        """
        Pick a random position within a circle of `radius`, 
        ensuring it doesn't overlap with existing objects.
        """
        o_pos = TDWUtils.array_to_vector3(
            TDWUtils.get_random_point_in_circle(center=np.array([0, 0, 0]), radius=radius)
        )
        ok = False
        count = 0
        while not ok and count < max_tries:
            count += 1
            ok = True
            for o in object_positions:
                if TDWUtils.get_distance(o.position, o_pos) <= o.radius:
                    ok = False
                    o_pos = TDWUtils.array_to_vector3(
                        TDWUtils.get_random_point_in_circle(center=np.array([0, 0, 0]), radius=radius)
                    )
        return o_pos


if __name__ == "__main__":
    # Example usage via command-line or direct call.
    args = get_args("ball_collisions")
    bd = BallsDataset()
    bd.run(
        num=args.num,        # Number of trials
        output_dir=args.dir, # Output directory
        temp_path=args.temp, # Temporary data path
        width=args.width,    # Render width
        height=args.height   # Render height
    )
