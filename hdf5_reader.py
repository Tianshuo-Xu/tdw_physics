# hdf5 reader
import h5py
import pickle

file = "/media/htung/Extreme SSD/fish/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_0/0000.pkl"

with open(file, "rb") as f:
    data = pickle.load(f)

"""
f["static"]
  - bounciness
  - color
  - distrators
  - dynamic_friction
  ....

# info for the inference frames before reset
f["static0"]


f["frames"]
  - 0000
     camera_matrices
     collision
     env_collisions
     images
        -_id
        -_img
     labels
     objects
  - 0001


"""

import ipdb; ipdb.set_trace()
import PIL.Image as Image
from PIL import ImageOps
import io
import os
import imageio

print("image size")
tmp = f["frames"]["0000"]["images"]["_img"][:]
image = Image.open(io.BytesIO(tmp))
image = ImageOps.mirror(image)
print("    ", image.size)


print("number of frames:", len(f["frames"]))

#input
# 1.5 secs after starting the frame
start_frame_id = f["static"]["start_frame_of_the_clip"][()] + 45

imgs = []
for img_id in range(start_frame_id):
    tmp = f["frames"][f"{img_id:04}"]["images"]["_img"][:]
    image = Image.open(io.BytesIO(tmp))
    image = ImageOps.mirror(image)
    imgs.append(image)

# make a video of the input
out = imageio.mimsave(
    os.path.join("tmp", 'input.gif'),
    imgs, fps = 30)

label = f["static"]["does_target_contact_zone"][()]

print("label", label)