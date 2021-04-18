import open3d as o3d
import ipdb
st=ipdb.set_trace

"""
using open3d to voxelize -> bad
"""

mesh_filename = "/home/htung/Documents/2021/example_meshes/input.obj"

mesh = o3d.io.read_triangle_mesh(mesh_filename)


o3d.visualization.draw_geometries([mesh])

voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=0.1)


st()

o3d.visualization.draw_geometries([voxel_grid])