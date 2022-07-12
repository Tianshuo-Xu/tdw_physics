from typing import List, Dict, Tuple, Optional

from tdw.controller import Controller
from tdw.add_ons.obi import Obi
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.obi_data.fluids.disk_emitter import DiskEmitter
from tdw.librarian import ModelRecord
from tdw_physics.util import MODEL_LIBRARIES

"""
Pour water into a receptacle.
"""

def add_transforms_object(record: ModelRecord,
                          position: Dict[str, float],
                          rotation: Dict[str, float],
                          o_id: Optional[int] = None,
                          add_data: Optional[bool] = True
) -> dict:
    """
    This is a wrapper for `Controller.get_add_object()` and the `add_object` command.
    This caches the ID of the object so that it can be easily cleaned up later.

    :param record: The model record.
    :param position: The initial position of the object.
    :param rotation: The initial rotation of the object, in Euler angles.
    :param o_id: The unique ID of the object. If None, a random ID is generated.
    :param add_data: whether to add the chosen data to the hdf5

    :return: An `add_object` command.
    """

    if o_id is None:
        o_id: int = Controller.get_unique_id()

    return {"$type": "add_object",
            "name": record.name,
            "url": record.get_url(),
            "scale_factor": record.scale_factor,
            "position": position,
            "rotation": rotation,
            "category": record.wcategory,
            "id": o_id}



def add_physics_object(record: ModelRecord,
                       position: Dict[str, float],
                       rotation: Dict[str, float],
                       mass: float,
                       dynamic_friction: float,
                       static_friction: float,
                       bounciness: float,
                       o_id: Optional[int] = None,
                       add_data: Optional[bool] = True
) -> List[dict]:
    """
    Get commands to add an object and assign physics properties. Write the object's static info to the .hdf5 file.

    :param o_id: The unique ID of the object. If None, a random ID will be generated.
    :param record: The model record.
    :param position: The initial position of the object.
    :param rotation: The initial rotation of the object, in Euler angles.
    :param mass: The mass of the object.
    :param dynamic_friction: The dynamic friction of the object's physic material.
    :param static_friction: The static friction of the object's physic material.
    :param bounciness: The bounciness of the object's physic material.
    :param add_data: whether to add the chosen data to the hdf5

    :return: A list of commands: `[add_object, set_mass, set_physic_material]`
    """
    # Get the add_object command.
    add_object = add_transforms_object(o_id=o_id,
                                            record=record,
                                            position=position,
                                            rotation=rotation,
                                            add_data=add_data
                                            )
    # Return commands to create the object.
    return [add_object,
            {"$type": "set_mass",
             "id": o_id,
             "mass": mass},
            {"$type": "set_physic_material",
             "id": o_id,
             "dynamic_friction": dynamic_friction,
             "static_friction": static_friction,
             "bounciness": bounciness}]


c = Controller()
c.communicate(Controller.get_add_scene(scene_name="tdw_room"))
camera = ThirdPersonCamera(position={"x": -1.75, "y": 0.6, "z": -0.5},
                           look_at={"x": 0, "y": 0.5, "z": 0})
obi = Obi()

c.add_ons.extend([camera, obi])
from tdw.obi_data.fluids.fluid import Fluid, FLUIDS
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.obi_data.fluids.disk_emitter import DiskEmitter


vis = [3.0, 3.0, 1.2, 0.01, 1.0]
fluid = Fluid(
capacity=1500,
resolution=1.0,
color={'a': 0.5, 'b': 0.995, 'g': 0.2, 'r': 0.2},
#color={'a': 1.0, 'b': 0.38, 'g': 0, 'r': 0.4},
rest_density=1000.0,
radius_scale=1.6, #2.0
random_velocity=vis[3],
smoothing=vis[0], #3.5, #3.0 #2.0 is like water, higher means stickier
surface_tension=1.0,
viscosity= vis[1], #0.001, #1.5
vorticity=vis[4], #0.7
reflection=0.25, ######0.25
transparency=0.85, #0.85, #####0.2
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
render_smoothness=0.8, #######0.8
metalness=0,
ambient_multiplier=1,
absorption=5,
refraction_downsample=1,
foam_downsample=1,
)
obi.create_fluid(fluid=fluid, #"honey",
                 shape=DiskEmitter(),
                 object_id=Controller.get_unique_id(),
                 position={"x": 0, "y": 2.35, "z": -0.3},
                 rotation={"x": 2, "y": 0, "z": 0},
                 speed=1.0)



# c.communicate(Controller.add_physics_object(model_name="cube",
#                                             object_id=Controller.get_unique_id(),
#                                             library="models_flex.json",
#                                             kinematic=True,
#                                             gravity=False,
#                                             scale_factor={"x": 0.5, "y": 0.5, "z": 0.5}))


record = [r for r in MODEL_LIBRARIES['models_flex.json'].records if r.name=="cube"][0]

commands = []
o_id= Controller.get_unique_id()
commands.extend(add_physics_object(record=record,
                                            position={'x':0, 'y':0, 'z':0},
                                            rotation={'x':0, 'y':0, 'z':0},
                                            o_id=o_id,
                                            mass= 1.0,
                                            dynamic_friction=0.4, #increased friction
                                            static_friction=0.4,
                                            bounciness=0))


rgb = [0,0,0]
scale = {'x': 0.5, 'y':0.5, 'z':0.5}
commands.extend([
    {"$type": "set_color",
     "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
     "id": o_id},
    {"$type": "scale_object",
     "scale_factor": scale,
     "id": o_id},
    {"$type": "set_kinematic_state",
             "id": o_id,
             "is_kinematic": True,
             "use_gravity": True}])

c.communicate(commands)
for i in range(500):
    c.communicate([])
c.communicate({"$type": "terminate"})