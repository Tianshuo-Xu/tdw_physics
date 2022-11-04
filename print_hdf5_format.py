# hdf5 reader
import h5py

#file = "/media/htung/Extreme SSD/fish/tdw_physics/dump/dominoes_origin/0003.hdf5"
#mp4_filename = "/media/htung/Extreme SSD/fish/tdw_physics/dump/dominoes_origin/0003_img.mp4"


file = "/media/htung/Extreme SSD/fish/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_0/0000_000.hdf5"
mp4_filename = "/media/htung/Extreme SSD/fish/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_0/0000_000_img.mp4"

import skvideo.io
videodata = skvideo.io.vread(mp4_filename)
print(videodata.shape)

##for i, im in enumerate(vid):
#    image = vid.get_data(num)

f = h5py.File(file, 'r')

"""
f["static"]
  - bounciness
  - color
  - distrators
  - dynamic_friction
  ....


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

import PIL.Image as Image
from PIL import ImageOps
import io
print("image size")
tmp = f["frames"]["0000"]["images"]["_img"][:]
image = Image.open(io.BytesIO(tmp))
image = ImageOps.mirror(image)
print("    ", image.size)


print("number of frames:", len(f["frames"]))

import ipdb; ipdb.set_trace()
print("hello")
