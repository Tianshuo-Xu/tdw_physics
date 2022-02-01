from time import sleep
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from tdw.output_data import OutputData, Images
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture

"""
Add a box and make it red.
"""

# # Add a camera and look at the object (we haven't added the object yet but this will execute after adding the object).
# cam = ThirdPersonCamera(position={"x": 2, "y": 1.6, "z": -0.6},
#                         look_at=object_id, avatar_id="a")
# c.add_ons.append(cam)
#
# cam_2 = ThirdPersonCamera(position={"x": 2, "y": 0.2, "z": -0.6},
#                         look_at=object_id, avatar_id="b")
#
# c.add_ons.append(cam_2)

#Save path
# path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("image_capture")
# capture = ImageCapture(path=path, avatar_ids=["a", "b"], pass_masks=["_img", "_id"])
# c.add_ons.append(capture)
# print(f"Images will be save to: {path.resolve()}")

c = Controller()

# Generate a unique object ID.
object_id = c.get_unique_id()

commands = [TDWUtils.create_empty_room(12, 12),
               c.get_add_object(model_name="iron_box",
                                library="models_core.json",
                                position={"x": 1, "y": 0, "z": -0.5},
                                object_id=object_id),
               {"$type": "set_color",
                "color": {"r": 1.0, "g": 0, "b": 0, "a": 1.0},
                "id": object_id}]

commands.extend(TDWUtils.create_avatar(position={"x": 2, "y": 1.6, "z": -0.6},
                                       avatar_id="a",
                                       look_at={"x": 0, "y": 0, "z": 0}))

commands.extend(TDWUtils.create_avatar(position={"x": 2, "y": 1.6, "z": -0.6},
                                       avatar_id="b",
                                       look_at={"x": 1, "y": 0, "z": -0.5}))

commands.extend([{"$type": "set_pass_masks",
                  "pass_masks": ["_img", "_id"],
                  "avatar_id": "a"},
          {"$type": "set_pass_masks",
                  "pass_masks": ["_img", "_id"],
                  "avatar_id": "b"},
                 {"$type": "send_images",
                  "frequency": "always",
                  "ids": ["a", "b"]}])

# Create the scene, add the object, and make the object red.
resp = c.communicate(commands)

output_directory = str(EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("send_images_2").resolve())
print(f"Images will be saved to: {output_directory}")

sleep(2)

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    # Get Images output data.
    if r_id == "imag":
        images = Images(resp[i])
        # Determine which avatar captured the image.
        if images.get_avatar_id() == "a":
            TDWUtils.save_images(images=images, filename="0", output_directory=output_directory)
        if images.get_avatar_id() == "b":
            TDWUtils.save_images(images=images, filename="1", output_directory=output_directory)

c.communicate({"$type": "terminate"})

# print("resp", resp.keys())