import h5py
import imageio
import os
import io
import PIL.Image as Image
from PIL import ImageOps
import imageio

def mkdir(path, is_assert=False):
    if is_assert:
        assert(not os.path.exists(path)), f"{path} exists, delete it first if you want to overwrite"
    if not os.path.exists(path):
        os.makedirs(path)


original_folder = "/mnt/fs4/hsiaoyut/tdw_physics/data"
output_folder = "/mnt/fs4/hsiaoyut/tdw_physics/small_train"
mode = "train"
scenarios = ["dominoes", "collision", "linking", "drop", "towers", "rollingSliding", "containment", "clothSagging"]

import ipdb; ipdb.set_trace()
for scenario in scenarios:
    arg_names = os.listdir(os.path.join(original_folder, scenario))
    arg_names = [arg for arg in arg_names if arg!="tfrecords"]
    import ipdb; ipdb.set_trace()
    for arg_name in arg_names:
        arg_full_path = os.path.join(original_folder, scenario, arg_name, mode)


        for trial_id in range(10):
            filename = os.path.join(arg_full_path, f"{trial_id:04}.hdf5")
            f = h5py.File(filename, "r")
            nsteps = len(f["frames"])
            output_dir =  os.path.join(output_folder, scenario, arg_name, mode)
            mkdir(output_dir)
            imgs = []
            for step in range(nsteps):
                tmp = f["frames"][f"{step:04}"]["images"]["_img"][:]
                rgbR = Image.open(io.BytesIO(tmp))
                imgs.append(rgbR)

            out = imageio.mimsave(
                os.path.join(output_dir, f'{trial_id:04}.gif'),
                imgs, fps = 20)


