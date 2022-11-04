# hdf5 reader
import h5py
import ast
import json
import numpy as np
import pickle
file = "/media/htung/Extreme SSD/fish/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_0/0002.hdf5"

f = h5py.File(file, 'r')
#data = f.get('static')[...].tolist()
#data2 = ast.literal_eval(data)
import ipdb; ipdb.set_trace()

def dfs_hdf5(hf, data):
  for attr in hf:
    data_ = hf[attr]
    if isinstance(data_, h5py.Group):
        data[attr] = dict()
        dfs_hdf5(data_, data[attr])
    else:
        data[attr] = data_[()]

output = dict()
dfs_hdf5(f, output)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (bytes)):
            return obj.decode('utf-8')
        print(obj)
        return json.JSONEncoder.default(self, obj)



json_str = json.dumps(output, indent = 4, cls=NpEncoder)
output_file = file.replace("0000.hdf5", "0000.json")
with open(output_file, "w") as outfile:
    outfile.write(json_str)

output_file = file.replace("0000.hdf5", "0000.pkl")
with open(output_file, 'wb') as outfile:
    pickle.dump(output, outfile)
import ipdb; ipdb.set_trace()
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