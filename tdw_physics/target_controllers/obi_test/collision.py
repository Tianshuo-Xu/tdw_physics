from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera


import os
gpu = 1
os.environ["DISPLAY"] = ":" + str(gpu)
#import ipdb; ipdb.set_trace()
c = Controller(launch_build=True)
camera = ThirdPersonCamera(position={"x": -3, "y": 2.1, "z": 0.5},
                           look_at={"x": 0, "y": 0, "z": 0})
c.add_ons.append(camera)
commands = [TDWUtils.create_empty_room(12, 12)]
commands.extend(c.get_add_physics_object(model_name="iron_box",
                                         object_id=c.get_unique_id(),
                                         position={"x": 0, "y": 3, "z": 0}))
commands.extend(c.get_add_physics_object(model_name="rh10",
                                         object_id=c.get_unique_id(),
                                         position={"x": 0, "y": 0, "z": 0}))
c.communicate(commands)
for i in range(200):
    c.communicate([])
c.communicate({"$type": "terminate"})