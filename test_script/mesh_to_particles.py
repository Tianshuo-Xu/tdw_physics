import trimesh
import copy
import numpy as np
#import pyvoxsurf
import os

import ipdb
st = ipdb.set_trace

"""
try to replicate the function from pyflex
https://github.com/YunzhuLi/PyFleX/blob/1391799c77b0bd8f5ab5a2a01f1f2ee64fb5a929/bindings/helpers.h#L154

"""


filename = "/home/htung/Documents/2021/example_meshes/0000_obj1_1.binvox"
import tdw_physics.binvox_rw as binvox_rw

with open(filename, 'rb') as f:
     m1 = binvox_rw.read_as_3d_array(f)


st()

# for visualization
axis = trimesh.creation.axis(axis_length=1)

# the size of the voxel grid in real-world scale
spacing = 0.02 # what is the distance between 2 particles

mesh_filename = "/home/htung/Documents/2021/example_meshes/0000_obj1.obj"
#tmp_mesh_fname = "/home/htung/Documents/2021/tmp/input.obj"
tmp_binvox_filename = "tmp.binvox"
mesh = trimesh.load_mesh(mesh_filename)
# make the mesh transparent
mesh.visual.face_colors[:,3] = 120
#(axis + mesh).show()

edges = mesh.bounding_box.extents
maxEdge = max(edges)
meshLower0 = mesh.bounds[0,:]
meshUpper0 = mesh.bounds[1,:]

# shift the mesh to it is in some bounding box [0, +x], [0, +y], [0, z]
mesh.vertices -= meshLower0
(axis + mesh).show()


edges = mesh.bounding_box.extents
maxEdge = max(edges)
meshLower = mesh.bounds[0,:]
meshUpper = mesh.bounds[1,:]
#  tweak spacing to avoid edge cases for particles laying on the boundary
# just covers the case where an edge is a whole multiple of the spacing.
spacingEps = spacing*(1.0 - 1e-4)
spacingEps_p = (9e-4)


# make sure to have at least one particle in each dimension
# naming is confusing, dx denotes the number of voxels in each dimension
dx = 1 if spacing > edges[0] else int(edges[0]/spacingEps)
dy = 1 if spacing > edges[1] else int(edges[1]/spacingEps)
dz = 1 if spacing > edges[2] else int(edges[2]/spacingEps)

maxDim = max(max(dx, dy), dz);

#expand border by two voxels to ensure adequate sampling at edges
# extending by a small offset to avoid point sitting exactly on the boundary
meshLower_spaced = meshLower - 2.0 * spacing - spacingEps_p
meshUpper_spaced = meshUpper +  2.0 * spacing + spacingEps_p

maxDim_spaced = maxDim + 4


# we shift the voxelization bounds so that the voxel centers
# lie symmetrically to the center of the object. this reduces the
# chance of missing features, and also better aligns the particles
# with the mesh
# ex. |1|1|1|0.3| --> |0.15|1|1|0.15|
meshOffset = np.zeros((3))
meshOffset[0] = 0.5 * (spacing - (edges[0] - (dx-1)*spacing))
meshOffset[1] = 0.5 * (spacing - (edges[1] - (dy-1)*spacing))
meshOffset[2] = 0.5 * (spacing - (edges[2] - (dz-1)*spacing))
meshLower_spaced -= meshOffset;

# original space
meshLower_spaced += meshLower0
meshUpper_spaced = meshLower_spaced + maxDim_spaced * spacing + 2 * spacingEps_p






#print(meshLower_spaced, meshUpper_spaced)
#print(f'binvox -aw -dc -d {maxDim_spaced} -pb -bb {meshLower_spaced[0]} {meshLower_spaced[1]} {meshLower_spaced[2]} {meshUpper_spaced[0]} {meshUpper_spaced[1]} {meshUpper_spaced[2]} -t binvox {mesh_filename}')
os.system(f'binvox -aw -dc -d {maxDim_spaced} -pb -bb {meshLower_spaced[0]} {meshLower_spaced[1]} {meshLower_spaced[2]} {meshUpper_spaced[0]} {meshUpper_spaced[1]} {meshUpper_spaced[2]} -t binvox {mesh_filename}')
print(meshLower_spaced, meshUpper_spaced)

# binvox -aw -dc -d 5 -pb -bb -0.9 -0.4 -0.9 0.9 1.4 0.9 -t binvox {mesh_filename}
#os.system(f"binvox -aw -dc -d 5 -pb -bb -0.9 -0.4 -0.9 0.9 1.4 0.9 -t binvox {mesh_filename}")
st()
voxelgrid = trimesh.voxel.creation.voxelize(mesh, spacing)

voxelgrid.show()




# add transformation
# add rotation
