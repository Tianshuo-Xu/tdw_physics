from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.object_manager import ObjectManager

"""
Bounce a ball on a table.
"""

c = Controller()
camera = ThirdPersonCamera(position={"x": 3, "y": 2.5, "z": -1},
                           look_at={"x": 0, "y": 0, "z": 0})
object_manager = ObjectManager(transforms=False, bounds=True, rigidbodies=True)
c.add_ons.extend([camera, object_manager])

commands = [TDWUtils.create_empty_room(12, 12)]
height = 500
width = 500
commands.extend([{"$type": "set_screen_size",	
	"width": width,
	"height": height}])


scale = 0.5
z = 0.2
y = 0

cube1_id = c.get_unique_id()
# Add a cube. Note that this is from the models_special.json model library.
commands.extend(c.get_add_physics_object(model_name="prim_cube",
                                    library="models_special.json",
                                    position={"x": 0.0, "y": y, "z": z},
                                    scale_factor={"x": scale, "y": scale, "z": scale},
                                    default_physics_values=False,
                                    mass=10,
                                    dynamic_friction=0.3,
                                    static_friction=0.3,
                                    bounciness=0.7,
                                    object_id=cube1_id))

cube2_id = c.get_unique_id()
# Add another cube. Note that this is from the models_special.json model library.
commands.extend(c.get_add_physics_object(model_name="prim_cube",
                                    library="models_special.json",
                                    position={"x": 0.0, "y": y, "z": -z},
                                    scale_factor={"x": scale, "y": scale, "z": scale},
                                    default_physics_values=False,
                                    mass=10,
                                    dynamic_friction=0.3,
                                    static_friction=0.3,
                                    bounciness=0.7,
                                    object_id=cube2_id))

# Re-initialize the object manager.
object_manager.initialized = False
c.communicate(commands)

# Wait until the ball stops moving.
while True:
    c.communicate([])
c.communicate({"$type": "terminate"})