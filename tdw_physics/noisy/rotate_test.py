from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.object_manager import ObjectManager
from tdw.output_data import OutputData, Rigidbodies

scale = 0.5
z = 0.1
y = 1
x = 0

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

obj1_id = c.get_unique_id()
# Add a cube. Note that this is from the models_special.json model library.
commands.extend(c.get_add_physics_object(model_name="prim_cube",
                                    library="models_special.json",
                                    position={"x": x, "y": y, "z": z},
                                    rotation = {"x":0, "y":0, "z":45},
                                    scale_factor={"x": scale, "y": scale, "z": scale},
                                    default_physics_values=False,
                                    mass=10,
                                    dynamic_friction=0.3,
                                    static_friction=0.3,
                                    bounciness=0.7,
                                    object_id=obj1_id))

# Re-initialize the object manager.
object_manager.initialized = False

c.communicate(commands)

# Wait until...
while True:
    resp = c.communicate([])
c.communicate({"$type": "terminate"})