# read out the obi stored states

import h5py
filename = "/media/htung/Extreme SSD/fish/tdw_physics/dump/drop/pilot_it2_drop_all_bowls_box/0001_001.hdf5"
import numpy as np
import imageio
import vispy.scene
import os
from vispy import app
from vispy.visuals import transforms

def add_floor(v):
    # add floor
    floor_thickness = 0.025
    floor_length = 8.0
    w, h, d = floor_length, floor_length, floor_thickness
    b1 = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
    #y_rotate(b1)
    v.add(b1)

    # adjust position of box
    mesh_b1 = b1.mesh.mesh_data
    v1 = mesh_b1.get_vertices()
    c1 = np.array([0., -floor_thickness*0.5, 0.], dtype=np.float32)
    mesh_b1.set_vertices(np.add(v1, c1))

    mesh_border_b1 = b1.border.mesh_data
    vv1 = mesh_border_b1.get_vertices()
    cc1 = np.array([0., -floor_thickness*0.5, 0.], dtype=np.float32)
    mesh_border_b1.set_vertices(np.add(vv1, cc1))


# initialize the scene
c = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white', size=(500, 300))
view = c.central_widget.add_view()
#c.show()
view.camera = vispy.scene.cameras.TurntableCamera(fov=50, azimuth=250, elevation=20, distance=6.0, up='+y')
add_floor(view)
p1 = vispy.scene.visuals.Markers()
p1.antialias = 0  # remove white edge
view.add(p1)
render_imgs = []
render_paths = []
#img = c.render()
# c.show()
f = h5py.File(filename, "r")
nsteps = len(f["frames"])
output_dir = "tmp"


for step_id in range(nsteps):
    step_data = f["frames"][f"{step_id:04}"]

    obi_object_ids = step_data["objects"]['obi_object_ids'][:].tolist()
    if len(obi_object_ids) > 0:
        particle_agg = []
        for object_id in obi_object_ids:
            particle_positions = step_data["objects"]["particles_positions"][f'{object_id}'][:]
            print(step_id, object_id, particle_positions.shape)
            particle_agg.append(particle_positions)

        particle_agg = np.concatenate(particle_agg, axis=0)
        colors = np.ones((particle_agg.shape[0], 4))
        p1.set_data(particle_positions, edge_color='black', face_color=colors, size=3)#, edge_width=0.02)
    img = c.render()
    out_filename = os.path.join(output_dir, f"{step_id}.png")
    vispy.io.write_png(out_filename, img)
    img = imageio.imread(out_filename)
    render_imgs.append(img)
    render_paths.append(out_filename)

out = imageio.mimsave(
    os.path.join(output_dir, 'vispy.gif'),
    render_imgs, fps = 20)

[os.remove(gt_path) for gt_path in render_paths]

