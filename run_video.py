# dump videos

import os
import h5py
import PIL.Image as Image
from PIL import ImageOps
import io
import imageio
path = "/mnt/fs4/hsiaoyut/tdw_physics/data/rollingSliding/pilot_it2_rollingSliding_simple_ledge_box/train"


for step in range(2):
    full_hdf5_path = os.path.join(path, f"{step:04}.hdf5")
    f = h5py.File(full_hdf5_path, "r")

    gt_imgs = []
    for step_ in range(100):
        tmp = f["frames"][f"{step_:04}"]["images"]["_img"][:]
        #seg_tmp = f["frames"][f"{step:04}"]["images"]["_id"]



        image = Image.open(io.BytesIO(tmp))
        image = ImageOps.mirror(image)
        gt_imgs.append(image)
    out = imageio.mimsave(f'tmp/{step}.gif', gt_imgs, fps = 20)