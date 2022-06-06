from tdw.controller import Controller
from tdw.add_ons.obi import Obi
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.obi_data.fluids.disk_emitter import DiskEmitter
from tdw.obi_data.fluids.cube_emitter import CubeEmitter
from tdw.obi_data.fluids.edge_emitter import EdgeEmitter
from tdw_physics.util import MODEL_LIBRARIES
from tdw.obi_data.fluids.fluid import Fluid, FLUIDS

import numpy as np
"""

Pour water into a receptacle.
"""
## dict_keys(['water', 'ink', 'oil', 'honey', 'glycerin', 'chocolate'])

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
camera = ThirdPersonCamera(position={"x": -2.75, "y": 1.5, "z": -0.5},
                           look_at={"x": 0, "y": 0, "z": 0})
obi = Obi()
c.add_ons.extend([camera, obi])

# smoothing, viscoity, thicknes, random velocuty, vorticity
vis = [3.0, 3.0, 0.001, 0.01, 1.0] #super sticky, and doesn't fall
#vis = [2.5, 0.05, 0.001, 0.01, 1.0] #super sticky, and doesn't fall

#vis = [2.0, 0.001, 1.2]
# vis = [2.0, 0.001, 0.001, 0.01, 1.0] #water
#vis = [1.5, 0.00001, 0.001, 0.01, 1.0] #water
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
                 position={"x": 0, "y": 1.5, "z": 0}, # y is height
                 rotation={"x": 100, "y": 0, "z": 0},
                 lifespan=1000,
                 speed=2)

PRIMITIVE_NAMES = [r.name for r in MODEL_LIBRARIES['models_special.json'].records if not r.do_not_use]
print(PRIMITIVE_NAMES)
record = [r for r in MODEL_LIBRARIES['models_special.json'].records if r.name=="prim_cube"][0]


o_id = Controller.get_unique_id()
# c.communicate({"$type": "add_object",
#                 "name": record.name,
#                 "url": record.get_url(),
#                 "scale_factor": {"x": 2, "y": 2, "z": 2},
#                 "category": record.wcategory,
#                 "id": o_id})

# c.communicate(
#                 {"$type": "set_kinematic_state",
#              "id": o_id,
#              "is_kinematic": True,
#              "use_gravity": False}
#     )

# c.communicate(Controller.get_add_physics_object(#model_name="prim_cube",
#                                                 model_name="triangular_prism",
#                                                 #model_name='trapezoidal_table',
#                                                 object_id=o_id,
#                                                 #library="models_full.json",
#                                                 #library="models_special.json",
#                                                 library="models_flex.json",
#                                                 kinematic=True,
#                                                 gravity=False,
#                                                 #position= {"x": 0, "y": 0.15, "z": 0},
#                                                 #scale_factor={"x": 0.5, "y": 0.5, "z": 0.5},
#                                                 position= {"x": 0, "y": 0.15, "z": 0},
#                                                 rotation= {"x": 135, "y": 0.0, "z": 0},
#                                                 scale_factor={"x": 0.5, "y": 0.5, "z": 0.5*np.sqrt(2)},
#                                                 default_physics_values=False))


c.communicate(Controller.get_add_physics_object(model_name="pyramid", #"prim_cube",
                                                object_id=o_id,
                                                library="models_flex.json",#"models_special.json",
                                                kinematic=True,
                                                gravity=False,
                                                position= {"x": 0, "y": 0.1, "z": 0.5},
                                                scale_factor={"x": 2.4, "y": 0.2, "z": 2.4},
                                                default_physics_values=False))
color = [0.3, 0.3, 0.3]
c.communicate(
            {"$type": "set_color",
             "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.},
             "id": o_id}
    )
for i in range(300):

    if i == 100:
        obi.set_fluid_speed(f_id, speed=0)

    # if i == 1000:
    #     obi.set_fluid_speed(f_id, speed=3.0)
    # if i == 1200:
    #     obi.set_fluid_speed(f_id, speed=0)
    c.communicate([])



commands = []
for oid in [o_id]:
    commands.append({"$type": "destroy_object",
                     "id": int(oid)})
c.communicate(commands)
obi.reset()



new_fid= f_id  #f_id + 5

obi.create_fluid(object_id = new_fid,
                 fluid=fluid,
                 shape=DiskEmitter(),
                 position={"x": 0, "y": 1.5, "z": 0}, # y is height
                 rotation={"x": 100, "y": 0, "z": 0},
                 lifespan=1000,
                 speed=2)

c.communicate(Controller.get_add_physics_object(model_name="pyramid", #"prim_cube",
                                                object_id=o_id,
                                                library="models_flex.json",#"models_special.json",
                                                kinematic=True,
                                                gravity=False,
                                                position= {"x": 0, "y": 0.1, "z": 0.5},
                                                scale_factor={"x": 1.2, "y": 0.2, "z": 1.2},
                                                default_physics_values=False))

for i in range(600):

    if i == 100:
        obi.set_fluid_speed(new_fid, speed=0)
    c.communicate([])

c.communicate({"$type": "terminate"})