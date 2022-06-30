from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.obi import Obi
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.obi_data.cloth.cloth_material import CLOTH_MATERIALS
from tdw.obi_data.cloth.cloth_material import ClothMaterial





cloth_material_names = [v for k, v in enumerate(CLOTH_MATERIALS)]

run_id = 6
print("cloth material type", cloth_material_names[run_id])
for k, v in CLOTH_MATERIALS[cloth_material_names[run_id]].__dict__.items():
    print(k, v)


# cloth material type silk
# visual_material cotton_jean_light_blue
# texture_scale {'x': 4, 'y': 4}
# visual_smoothness 0
# stretching_scale 1.0
# stretch_compliance 0
# max_compression 0
# max_bending 0.05
# bend_compliance 0
# drag 0.05
# lift 0.05
# mass_per_square_meter 0.136


# cloth material type plastic
# visual_material plastic_vinyl_glossy_white
# texture_scale {'x': 1, 'y': 1}
# visual_smoothness 0
# stretching_scale 0.75 # too large: cloth becomes large and deformable, small: cloth becomes small
# stretch_compliance 0
# max_compression 0
# max_bending 0.0 # large means the object will shrink
# bend_compliance 0
# drag 0.05
# lift 0.05
# mass_per_square_meter 0.15



c = Controller()
camera = ThirdPersonCamera(position={"x": -3.75, "y": 1.5, "z": -0.5},
                           look_at={"x": 0, "y": 1.25, "z": 0})
obi = Obi()
c.add_ons.extend([camera, obi])
# Create a cloth sheet.
cloth_material = ClothMaterial(visual_material="3d_printed_mesh_round",
                               texture_scale={"x": 1, "y": 1},
                               stretching_scale=1,
                               stretch_compliance=0.002,
                               max_compression=0.5,
                               max_bending=0.05,
                               bend_compliance=1.0,
                               drag=0.05,
                               lift=0.05,
                               visual_smoothness=0,
                               mass_per_square_meter=0.15)

# Fabric, Leather, Metal, Paper, Plastic
# cloth_material = ClothMaterial(visual_material="3d_printed_mesh_round",
#                                texture_scale={"x": 1, "y": 1},
#                                stretching_scale=0.75,
#                                stretch_compliance=0,
#                                max_compression=0,
#                                max_bending=0,
#                                drag=0.05,
#                                lift=0.05,
#                                visual_smoothness=0,
#                                mass_per_square_meter=0.15)



obi.create_cloth_sheet(cloth_material=cloth_material, #cloth_material_names[run_id],
                       object_id=Controller.get_unique_id(),
                       position={"x": 0, "y": 2, "z": 0},
                       rotation={"x": 0, "y": 0, "z": 0})
commands = [TDWUtils.create_empty_room(12, 12)]
commands.extend(Controller.get_add_physics_object(model_name="sphere",
                                                  object_id=Controller.get_unique_id(),
                                                  library="models_flex.json",
                                                  kinematic=True,
                                                  gravity=False,
                                                  scale_factor={"x": 0.5, "y": 0.5, "z": 0.5}))
c.communicate(commands)
# Let the cloth fall.
for i in range(150):
    c.communicate([])
c.communicate({"$type": "terminate"})