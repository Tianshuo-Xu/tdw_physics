from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.object_manager import ObjectManager
from tdw.output_data import OutputData, Rigidbodies

scale = 0.5
z = 0.1
y = 0.5
x = 0
use_settle = True

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
commands.extend(c.get_add_physics_object(model_name="prim_cyl",
                                    library="models_special.json",
                                    position={"x": x, "y": y, "z": z},
                                    scale_factor={"x": scale, "y": scale, "z": scale},
                                    default_physics_values=False,
                                    mass=10,
                                    dynamic_friction=0.3,
                                    static_friction=0.3,
                                    bounciness=0.7,
                                    object_id=obj1_id))

obj2_id = c.get_unique_id()
# Add another cube. Note that this is from the models_special.json model library.
commands.extend(c.get_add_physics_object(model_name="prim_cyl",
                                    library="models_special.json",
                                    position={"x": x, "y": y, "z": -z},
                                    scale_factor={"x": scale, "y": scale, "z": scale},
                                    default_physics_values=False,
                                    mass=10,
                                    dynamic_friction=0.3,
                                    static_friction=0.3,
                                    bounciness=0.7,
                                    object_id=obj2_id))

# Re-initialize the object manager.
object_manager.initialized = False

if use_settle:
    commands.extend([{"$type": "set_kinematic_state",
                            "id": obj1_id,
                            "use_gravity": False},
                    {"$type": "set_kinematic_state",
                            "id": obj2_id,
                            "use_gravity": False},
                    {"$type": "set_time_step",
                            "time_step": 0.0001},
                    {"$type": "step_physics",
                            "frames": 500}])
    c.communicate(commands)

    commands = []
    commands.extend([{"$type": "set_time_step",
                            "time_step": 0.03},
                    {"$type": "set_kinematic_state",
                            "id": obj1_id,
                            "use_gravity": True},
                    {"$type": "set_kinematic_state",
                            "id": obj2_id,
                            "use_gravity": True}])

resp = c.communicate(commands)

# Wait until...
while True:
    for i in range(len(resp) - 1):
        r_id = OutputData.get_data_type_id(resp[i])
        if r_id == "rigi":
            rigi = Rigidbodies(resp[i])
            for j in range(rigi.get_num()):
                object_id = rigi.get_id(j)
                velocity = rigi.get_velocity(j)
                sleeping = rigi.get_sleeping(j)
                print(j, velocity)
    print('////////////////////////////////////////////////')
    resp = c.communicate([])
c.communicate({"$type": "terminate"})