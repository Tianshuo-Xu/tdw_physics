# hdf5 reader
import os
import h5py
import pickle
import cv2
import imageio
import math
import skvideo.io
import numpy as np
import trimesh

from scipy.spatial.transform import Rotation as R


def get_depth_values(image: np.array, depth_pass: str = "_depth", width: int = 256, height: int = 256, near_plane: float = 0.1, far_plane: float = 100) -> np.array:
    """
    Get the depth values of each pixel in a _depth image pass.
    The far plane is hardcoded as 100. The near plane is hardcoded as 0.1.
    (This is due to how the depth shader is implemented.)
    :param image: The image pass as a numpy array.
    :param depth_pass: The type of depth pass. This determines how the values are decoded. Options: `"_depth"`, `"_depth_simple"`.
    :param width: The width of the screen in pixels. See output data `Images.get_width()`.
    :param height: The height of the screen in pixels. See output data `Images.get_height()`.
    :param near_plane: The near clipping plane. See command `set_camera_clipping_planes`. The default value in this function is the default value of the near clipping plane.
    :param far_plane: The far clipping plane. See command `set_camera_clipping_planes`. The default value in this function is the default value of the far clipping plane.
    :return An array of depth values.
    """
    image = np.reshape(image, (height, width, 3))

    # Convert the image to a 2D image array.
    if depth_pass == "_depth":
        depth_values = np.array((image[:, :, 0] + image[:, :, 1] / 256.0 + image[:, :, 2] / (256.0 ** 2)))
    elif depth_pass == "_depth_simple":
        depth_values = image[:, :, 0]
    else:
        raise Exception(f"Invalid depth pass: {depth_pass}")
    # Un-normalize the depth values.
    return (depth_values * ((far_plane - near_plane) / 256.0)).astype(np.float32)

def get_intrinsics_from_projection_matrix(proj_matrix, size):
    H, W = size
    vfov = 2.0 * math.atan(1.0/proj_matrix[1][1]) * 180.0/ np.pi
    vfov = vfov / 180.0 * np.pi
    tan_half_vfov = np.tan(vfov / 2.0)
    tan_half_hfov = tan_half_vfov * H / float(H)
    fx = W / 2.0 / tan_half_hfov  # focal length in pixel space
    fy = H / 2.0 / tan_half_vfov

    pix_T_cam = np.array([[fx, 0, W / 2.0],
                           [0, fy, H / 2.0],
                                   [0, 0, 1]])
    return pix_T_cam

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def Pixels2Camera_np(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are B x H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    # there is no randomness here

    B, H, W = list(z.shape)

    fx = np.reshape(fx, [B,1,1])
    fy = np.reshape(fy, [B,1,1])
    x0 = np.reshape(x0, [B,1,1])
    y0 = np.reshape(y0, [B,1,1])

    # unproject
    EPS = 1e-6
    x = ((z+EPS)/fx)*(x-x0)
    y = ((z+EPS)/fy)*(y-y0)

    x = np.reshape(x, [B,-1])
    y = np.reshape(y, [B,-1])
    z = np.reshape(z, [B,-1])
    xyz = np.stack([x,y,z], axis=2)
    return xyz


def meshgrid2d_py(Y, X):
    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [Y, 1])
    grid_y = np.tile(grid_y, [1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, X])
    grid_x = np.tile(grid_x, [Y, 1])

    return grid_y, grid_x

def depth2pointcloud_np(z, pix_T_cam):
    B, C, H, W = list(z.shape)  # this is 1, 1, H, W
    y, x = meshgrid2d_py(H, W)
    y = np.repeat(y[np.newaxis, :, :], B, axis=0)
    x = np.repeat(x[np.newaxis, :, :], B, axis=0)
    z = np.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera_np(x, y, z, fx, fy, x0, y0)
    return xyz


camera_far_plane=100
camera_near_plane=0.1


scenario_name = "bouncy_platform_pp"
trial_name = "bouncy_platform-use_blocker_with_hole=0-target=bowl-tscale=0.15,0.15,0.15"
trial_id = 0
show_timestep = 5
file = f"/media/htung/Extreme SSD/fish/tdw_physics/dump/{scenario_name}/{trial_name}/{trial_id:04}.pkl"

png_video_file = f"/media/htung/Extreme SSD/fish/tdw_physics/dump/{scenario_name}/{trial_name}/{trial_id:04}_img.mp4"
id_video_file = f"/media/htung/Extreme SSD/fish/tdw_physics/dump/{scenario_name}/{trial_name}/{trial_id:04}_id.json"
depth_video_file = f"/media/htung/Extreme SSD/fish/tdw_physics/dump/{scenario_name}/{trial_name}/{trial_id:04}_depth.mp4"
#id_video_file = "/media/htung/Extreme SSD/fish/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_0/0000_id.json"
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



videodata = skvideo.io.vread(png_video_file)
print("video size:", videodata.shape[1:])
print("number of frames:", videodata.shape[0])

# 1. get rgb video input and label
#input
# 1.5 secs after starting the frame
#start_frame_id = data["static"]["start_frame_of_the_clip"] + 45
start_frame_id = data["static"]["start_frame_for_prediction"]
out = imageio.mimsave(
    os.path.join("tmp", 'input.gif'),
    videodata[:start_frame_id], fps = 30)

label = data["static"]["does_target_contact_zone"]
print("label", label)


# 2. get depth input -- check by forming point cloud from depth, and 3d reconstrution from mesh
use_video = False
timestep = show_timestep
if use_video:
    depth_videodata = skvideo.io.vread(depth_video_file)
    #imageio.imwrite("depth_video.png", depth_videodata[0])
    T, W, H, C = depth_videodata.shape
else:
    depth_videodata = None
    depth_img_file = f"/media/htung/Extreme SSD/fish/tdw_physics/dump/{scenario_name}/{trial_name}/{trial_id:04}_depth/depth_{show_timestep:04}.png"
    depth_img = imageio.imread(depth_img_file)
    W,H,C = depth_img.shape
    imageio.imwrite("depth.png", depth_img)


depth = get_depth_values(image=depth_videodata[timestep] if use_video else depth_img, width=W, height=H\
                                    ,far_plane=camera_far_plane, near_plane=camera_near_plane)


imageio.imwrite("depth.png", depth)
proj_matrix = np.reshape(data["frames"][f"{timestep:04}"]["camera_matrices"]["projection_matrix"], [4,4])
pix_T_cam = get_intrinsics_from_projection_matrix(proj_matrix, size=depth.shape)


pts_cam = depth2pointcloud_np(depth[np.newaxis, np.newaxis, :, :], pix_T_cam[np.newaxis, :, :])[0]
pts_cam = pts_cam[pts_cam[:,2] < 10]
axis = trimesh.creation.axis(axis_length=1)


positions = data["frames"][f"{timestep:04}"]["objects"]["positions"]
rotations = data["frames"][f"{timestep:04}"]["objects"]["rotations"]#x,y,z,w


obj_meshes = []
object_ids = data["static"]["object_ids"]
for idx, object_id in enumerate(object_ids):
    vertices = data["static"]["mesh"][f"vertices_{idx}"]
    faces = data["static"]["mesh"][f"faces_{idx}"]
    mesh = trimesh.Trimesh(vertices=vertices * data["static"]["scale"][idx],
                       faces=faces)
    rot = np.eye(4)
    rot[:3, :3] = R.from_quat(rotations[idx]).as_matrix()
    mesh.apply_transform(rot)
    mesh.apply_translation(positions[idx])
    mesh.visual.face_colors = [255, 100, 100, 255]
    obj_meshes.append(mesh)

camera_matrix = np.reshape(data["frames"][f"{timestep:04}"]["camera_matrices"]['camera_matrix'], [4,4])
world_T_cam = np.linalg.inv(camera_matrix)
import ipdb; ipdb.set_trace()
camera = trimesh.creation.box(extents=(0.2,0.1,0.08), transform=world_T_cam)
(sum(obj_meshes) + camera + axis).show()
#import ipdb; ipdb.set_trace()

colors = np.zeros((pts_cam.shape[0], 4), dtype=np.uint8)
colors[:,3] = 255
colors[:,0] = 255

pcd = trimesh.PointCloud(pts_cam, colors=colors)
(trimesh.Scene(pcd) + axis).show()


# 3. get segmentation input
#id_videodata = skvideo.io.vread(id_video_file)
import json
from pycocotools import mask as cocomask

seg_colors = data["static"]["video_object_segmentation_colors"]
nobjs = seg_colors.shape[0]

with open(id_video_file, "r") as fh_read:
    in_dict = json.load(fh_read)


obj_info_dict = dict()
for timestep in range(start_frame_id):
    obj_info_dict[timestep] = dict()
    for obj_info in in_dict[f'{timestep:04}']:
        idx = obj_info["idx"]
        obj_info_dict[timestep][idx] = obj_info

#for timestep in range(start_frame_id)
for idx in range(nobjs):
    frames = []
    for timestep in range(start_frame_id):
        if idx in obj_info_dict[timestep]:
            rgn = cocomask.decode(obj_info_dict[timestep][idx])
            frames.append(rgn * 255)
        else:
            frames.append(np.zeros((H, W)))

    out = imageio.mimsave(
        os.path.join("tmp", f'id{idx}.gif'),
        frames, fps = 30)

