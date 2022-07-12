from tdw.controller import Controller
from tdw.add_ons.obi import Obi
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.obi_data.fluids.disk_emitter import DiskEmitter
from tdw.obi_data.fluids.cube_emitter import CubeEmitter
from tdw.obi_data.fluids.edge_emitter import EdgeEmitter
from tdw_physics.util import MODEL_LIBRARIES
from tdw.obi_data.fluids.fluid import Fluid, FLUIDS

import os
import numpy as np
"""

Pour water into a receptacle.
"""
## dict_keys(['water', 'ink', 'oil', 'honey', 'glycerin', 'chocolate'])



from tdw.obi_data.fluids.fluid import FLUIDS

fluid = FLUIDS["water"]
for k in fluid.__dict__:
    print(f'{k}={fluid.__dict__[k]}')

import ipdb; ipdb.set_trace()
class CustomModelLoader():
    def __init__(self, asset_dir):
        """
        root path for your model
        """
        self.asset_dir = os.path.abspath(asset_dir)
    def get_record(self, name):
        from json import loads
        from tdw.librarian import ModelRecord

        with open(os.path.join(self.asset_dir, name, "record.json"), 'r') as j:
            tmp = loads(j.read())
            tmp["urls"]["Linux"] = "file://" + os.path.join(self.asset_dir, f"{name}/StandaloneLinux64/{name}")
            #/home/htung/Documents/2021/tdw_physics/tdw_physics/target_controllers/asset/tri_slope/StandaloneLinux64/tri_slope"
        record = ModelRecord(tmp)
        return record
# blood
# chocolate
# corn_oil
# glycerin
# honey
# ink
# milk
# molasses
# motor_oil
# water
# gravel
# rocks
PRIMITIVE_NAMES = [r.name for r in MODEL_LIBRARIES['models_full.json'].records if not r.do_not_use]
print([r for r in PRIMITIVE_NAMES if "trapezoid" in r])

c = Controller()
c.communicate(Controller.get_add_scene(scene_name="tdw_room"))
camera = ThirdPersonCamera(position={"x": -2.75, "y": 1.5, "z": -0.57},
                           look_at={"x": 0, "y": 0, "z": 0})
obi = Obi()
c.add_ons.extend([camera, obi])

# smoothing, viscoity, thicknes, random velocuty, vorticity
#vis = [2.0, 3.0, 0.001, 0.01, 1.0] #super sticky, and doesn't fall
#vis = [3.0, 0.5, 0.001, 0.01, 1.0]
#vis = [3.0, 0.1, 0.001, 0.01, 1.0] #super sticky, and doesn't fall
#vis = [2.5, 0.01, 0.001, 0.01, 1.0] #super sticky, and doesn't fall

#vis = [2.0, 0.001, 1.2]
#vis = [2.0, 0.001, 0.001, 0.01, 1.0] #water
vis = [1.5, 0.00001, 0.001, 0.01, 1.0] #water

#vis = [1.5, 0.1, 0.001, 0.01, 1.0] #water
#note
# smoothing: don't set below 2.0, otherwise the liquid looks scattered
# vorticity,random velocity: doesn't look too different in our setup
# thickness is related to blending and not actual physics, set it to constant
fluid = Fluid(
capacity=1500,
resolution=1.0,
color={'a': 0.5, 'b': 0.995, 'g': 0.2, 'r': 0.2},
rest_density=1000.0,
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

# fluid = FLUIDS["honey"]
# for k in fluid.__dict__:
#     print(f'{k}={fluid.__dict__[k]}')
# import ipdb; ipdb.set_trace()
f_id = Controller.get_unique_id()
obi.create_fluid(object_id = f_id,
                 fluid=fluid,
                 shape=DiskEmitter(),
                 position={"x": 0, "y": 1.2, "z": 0.95}, # y is height
                 rotation={"x": 140, "y": 0, "z": 0},
                 lifespan=1000,
                 speed=2)


# obi.create_fluid(object_id = f_id,
#                  fluid=fluid,
#                  shape=DiskEmitter(),
#                  position={"x": 0, "y": 1.4, "z": 1.1}, # y is height
#                  rotation={"x": 130, "y": 0, "z": 0},
#                  lifespan=1000,
#                  speed=2)

PRIMITIVE_NAMES = [r.name for r in MODEL_LIBRARIES['models_special.json'].records if not r.do_not_use]
print(PRIMITIVE_NAMES)
record = [r for r in MODEL_LIBRARIES['models_flex.json'].records if r.name=="cube"][0]




cm_loader = CustomModelLoader("../asset")

# import ipdb; ipdb.set_trace()

# from json import loads
# from tdw.librarian import ModelRecord

# with open("../asset/tri_slope/record.json", 'r') as j:
#     tmp = loads(j.read())
#     tmp["urls"]["Linux"] = "file:///home/htung/Documents/2021/tdw_physics/tdw_physics/target_controllers/asset/tri_slope/StandaloneLinux64/tri_slope"

# import ipdb; ipdb.set_trace()
# record = ModelRecord(tmp)

record = cm_loader.get_record("tri_slope")
o_id = Controller.get_unique_id()

def add_custom_obj(record, o_id, position, rotation=None, scale_factor=None):
# add_custom_obj(record, o_id, rotation={"x": 0.0, "y": 90.0, "z": 0}, scale_factor={"x": 0.5, "y": 0.5, "z": 0.5})
    if position is None:
        position = {"x": 0.0, "y": 0, "z": 0}
    commands = []
    commands.extend([{"$type": "add_object",
                    "name": record.name,
                    "url": record.get_url(),
                    "scale_factor": record.scale_factor,
                    "position": position,
                    "category": record.wcategory,
                    "id": o_id}])
    commands.extend(
                    [{"$type": "set_kinematic_state",
                 "id": o_id,
                 "is_kinematic": True,
                 "use_gravity": False}
        ])
    commands.extend([{"$type": "set_object_collision_detection_mode",
                     "id": o_id,
                     "mode": "continuous_speculative"}])

    if rotation is not None:
        commands.extend([{"$type": "rotate_object_to_euler_angles",
                         "euler_angles": rotation,
                         "id": o_id}])
    if scale_factor is not None:
        commands.extend([{"$type": "scale_object_and_mass",
                         "scale_factor": scale_factor,
                         "id": o_id}])
    return commands
commands = []
commands = add_custom_obj(record,
                         o_id,
                         position={"x": 0.0, "y": 0.05, "z": -0.38},
                         rotation={"x": 0.0, "y": 90.0, "z": 0},
                         scale_factor={"x": 0.2, "y": 0.1, "z": 0.5})


#c.communicate(commands)
o_id = Controller.get_unique_id()
commands.extend(Controller.get_add_physics_object(#model_name="prim_cube",
                                                model_name="cube",
                                                #model_name='trapezoidal_table',
                                                object_id=o_id,
                                                #library="models_full.json",
                                                #library="models_special.json",
                                                library="models_flex.json",
                                                kinematic=True,
                                                gravity=False,
                                                #position= {"x": 0, "y": 0.15, "z": 0},
                                                #scale_factor={"x": 0.5, "y": 0.5, "z": 0.5},
                                                position= {"x": 0, "y": 0, "z": 0.7},
                                                rotation= {"x": 0.0, "y": 0.0, "z": 0},
                                                scale_factor={"x": 0.8, "y": 0.5, "z": 0.8},
                                                default_physics_values=False))

color = [0.3, 0.3, 0.3]
commands.extend([
            {"$type": "set_color",
             "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.},
             "id": o_id}]
    )

o_id = Controller.get_unique_id()
# commands.extend(add_custom_obj(record,
#                          o_id,
#                          position={"x": 0, "y": 0.5 + 0.3, "z": 0.5},
#                          rotation={"x": 0.0, "y": 90.0, "z": 0},
#                          scale_factor={"x": 0.5, "y": 0.3, "z": 0.5}))
commands.extend(add_custom_obj(record,
                         o_id,
                         position={"x": 0, "y": 0.5 + 0.23, "z": 0.5},
                         rotation={"x": 0.0, "y": 90.0, "z": 0},
                         scale_factor={"x": 0.5, "y": 0.23, "z": 0.5}))


c.communicate(commands)
# c.communicate(Controller.get_add_physics_object(model_name="pyramid", #"prim_cube",
#                                                 object_id=o_id,
#                                                 library="models_flex.json",#"models_special.json",
#                                                 kinematic=True,
#                                                 gravity=False,
#                                                 position= {"x": 0, "y": 0.1, "z": 0.5},
#                                                 scale_factor={"x": 2.4, "y": 0.2, "z": 2.4},
#                                                 default_physics_values=False))
import time
for i in range(1000):

    if i == 100:
        obi.set_fluid_speed(f_id, speed=0)
    time.sleep(0.1)
    c.communicate([])


c.communicate({"$type": "terminate"})