from typing import List
from tdw_physics.rigidbodies_dataset import RigidbodiesDataset
from tdw_physics.util import get_random_position

class DominoesSimulation(RigidbodiesDataset):
    def get_scene_initialization_commands(self) -> List[dict]:
        # Initialize the scene with a simple room
        return [self.get_add_scene(scene_name="tdw_room")]

    def get_trial_initialization_commands(self) -> List[dict]:
        commands = []
        # Define the dimensions and spacing for the dominoes
        domino_height = 0.2
        domino_width = 0.05
        domino_depth = 0.01
        spacing = 0.06  # Space between dominoes

        # Position the first domino
        x_position = 0.0
        z_position = 0.0

        for i in range(4):
            # Add each domino with a slight offset in the x direction
            position = {"x": x_position, "y": domino_height / 2, "z": z_position}
            rotation = {"x": 0, "y": 0, "z": 0}
            # Add the domino object
            commands.extend(self.add_physics_object(
                model_name="domino",
                position=position,
                rotation=rotation,
                mass=0.1,
                dynamic_friction=0.4,
                static_friction=0.6,
                bounciness=0.3
            ))
            # Update the x position for the next domino
            x_position += domino_width + spacing

        # Apply an initial force to the first domino to start the chain reaction
        force = {"x": 1.0, "y": 0.0, "z": 0.0}
        torque = {"x": 0.0, "y": 0.0, "z": 0.0}
        commands.append({
            "$type": "apply_force_to_object",
            "force": force,
            "torque": torque,
            "id": self.object_ids[0]
        })

        return commands

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:
        # No additional per-frame commands are needed
        return []

    def get_field_of_view(self) -> float:
        # Set the camera's field of view
        return 60.0

if __name__ == "__main__":
    # Create the simulation and run it
    sim = DominoesSimulation()
    sim.run(num_trials=1, output_dir="output/")

