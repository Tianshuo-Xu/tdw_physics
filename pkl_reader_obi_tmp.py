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
import io
from PIL import Image
import copy


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

def apply_4x4(RT, xyz):
    if isinstance(xyz, np.ndarray):
        append_bdim = False
        if len(xyz.shape) == 2:
            append_bdim = True
            xyz = xyz[np.newaxis,...]
            RT = RT[np.newaxis,...]

        B, N, _ = list(xyz.shape)
        ones = np.ones_like(xyz[:,:,0:1])
        xyz1 = np.concatenate([xyz, ones], 2)
        xyz1_t = np.transpose(xyz1, (0, 2, 1))
        # this is B x 4 x N
        xyz2_t = np.matmul(RT, xyz1_t)
        xyz2 = np.transpose(xyz2_t, (0, 2, 1))
        xyz2 = xyz2[:,:,:3]
        if append_bdim:
            xyz2 = xyz2[0]
        return xyz2
    else:
        B, N, _ = list(xyz.shape)
        ones = torch.ones_like(xyz[:,:,0:1])
        xyz1 = torch.cat([xyz, ones], 2)
        xyz1_t = torch.transpose(xyz1, 1, 2)
        # this is B x 4 x N
        xyz2_t = torch.matmul(RT, xyz1_t)
        xyz2 = torch.transpose(xyz2_t, 1, 2)
        xyz2 = xyz2[:,:,:3]
        return xyz2


def get_fov_from_intrinsic(pix_T_cam):
    w = pix_T_cam[0,2]*2
    h = pix_T_cam[1,2]*2
    fx = pix_T_cam[0,0]
    fy = pix_T_cam[1,1]
    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

    return (fov_x, fov_y)
def meshgrid2d_py(Y, X):
    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [Y, 1])
    grid_y = np.tile(grid_y, [1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, X])
    grid_x = np.tile(grid_x, [Y, 1])

    return grid_y, grid_x


def Camera2Pixels_np(xyz, pix_T_cam):
    # xyz is shaped N x 3
    # pix_T_cam: 3x3
    # returns xy, shaped B x H*W x 2

    fx, fy = pix_T_cam[0,0], pix_T_cam[1,1]
    x0, y0 = pix_T_cam[0,2], pix_T_cam[1,2]

    #x, y, z = np.unstack(xyz, dim=-1)
    x, y, z = np.split(xyz, 3, axis=-1)
    B = list(z.shape)[0]

    EPS = 1e-4
    z = np.maximum(z, EPS * np.ones_like(z))
    x = (x*fx)/z + x0
    y = (y*fy)/z + y0

    xy = np.concatenate([x, y], axis=-1)
    return xy


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
data_root = "/media/htung/Extreme SSD/fish/tdw_physics/dump_mini4/"


scenario_name = "mass_dominoes_pp"
postfix = "-copy0"
trial_name = "mass_dominoes-num_middle_objects=3-star_putfirst=0-remove_middle=0"
trial_id = 0
show_timestep = 50
file = os.path.join(data_root, f"{scenario_name}{postfix}/{trial_name}/{trial_id:04}.pkl")

png_video_file = os.path.join(data_root, f"{scenario_name}{postfix}/{trial_name}/{trial_id:04}_img.mp4")
id_video_file = os.path.join(data_root, f"{scenario_name}{postfix}/{trial_name}/{trial_id:04}_id.json")
depth_video_file = os.path.join(f"{scenario_name}{postfix}/{trial_name}/{trial_id:04}_depth.mp4")
#id_video_file = "/media/htung/Extreme SSD/fish/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_0/0000_id.json"
with open(file, "rb") as f:
    data = pickle.load(f)

use_obi = False
if "obi_object_ids" in data['static']:
    print("obi object ids")
    print( data['static']['obi_object_ids'])
    obi_object_ids = data['static']['obi_object_ids']
    obi_object_type = data['static']['obi_object_type']
    use_obi = data['static']["use_obi"]


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
_, H, W, C = videodata.shape

# 1. get rgb video input and label
#input
# 1.5 secs after starting the frame
#start_frame_id = data["static"]["start_frame_of_the_clip"] + 45
start_frame_id = data["static"]["start_frame_for_prediction"] + 100
out = imageio.mimsave(
    os.path.join("tmp", 'input.gif'),
    videodata[:start_frame_id], fps = 30)

label = data["static"]["does_target_contact_zone"]
print("label", label)


# 2. get depth input -- check by forming point cloud from depth, and 3d reconstrution from mesh
# use_video = False
timestep = show_timestep
# if use_video:
#     depth_videodata = skvideo.io.vread(depth_video_file)
#     #imageio.imwrite("depth_video.png", depth_videodata[0])
#     T, W, H, C = depth_videodata.shape
# else:
#     depth_videodata = None
#     depth_img_file = f"/media/htung/Extreme SSD/fish/tdw_physics/dump/{scenario_name}/{trial_name}/{trial_id:04}_depth/depth_{show_timestep:04}.png"
#     depth_img = imageio.imread(depth_img_file)
#     W,H,C = depth_img.shape
#     imageio.imwrite("depth.png", depth_img)


# depth = get_depth_values(image=depth_videodata[timestep] if use_video else depth_img, width=W, height=H\
#                                     ,far_plane=camera_far_plane, near_plane=camera_near_plane)


# imageio.imwrite("depth.png", depth)
# proj_matrix = np.reshape(data["frames"][f"{timestep:04}"]["camera_matrices"]["projection_matrix"], [4,4])
# pix_T_cam = get_intrinsics_from_projection_matrix(proj_matrix, size=depth.shape)


# pts_cam = depth2pointcloud_np(depth[np.newaxis, np.newaxis, :, :], pix_T_cam[np.newaxis, :, :])[0]
# pts_cam = pts_cam[pts_cam[:,2] < 10]
axis = trimesh.creation.axis(axis_length=1)


positions = data["frames"][f"{timestep:04}"]["objects"]["positions"]
rotations = data["frames"][f"{timestep:04}"]["objects"]["rotations"]#x,y,z,w


obj_meshes = []
obj_meshes_pts = []
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

    pts = mesh.vertices
    colors = np.zeros((pts.shape[0], 4), dtype=np.uint8)
    colors[:,3] = 255
    colors[:,1] = 255
    obj_meshes_pts.append(trimesh.PointCloud(pts, colors=colors))

# get camera extrinsic and intrinsic
cam_T_world = np.reshape(data["frames"][f"{timestep:04}"]["camera_matrices"]['camera_matrix'], [4,4])
world_T_cam = np.linalg.inv(cam_T_world)
rot = np.array([[-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]])
world_T_cam  = np.dot(world_T_cam , rot)
cam_T_world = np.linalg.inv(world_T_cam)

proj_matrix = np.reshape(data["frames"][f"{timestep:04}"]["camera_matrices"]["projection_matrix"], [4,4])
pix_T_cam = get_intrinsics_from_projection_matrix(proj_matrix, size=(H,W))

camera = trimesh.creation.box(extents=(0.2,0.1,0.08), transform=world_T_cam)

#scene = trimesh.scene.Scene()
#for obj_mesh in obj_meshes:
#    scene.add_geometry(obj_mesh)

trans = np.eye(4)
trans[:3,3] = np.array([0,-0.05, 0])
floor = trimesh.creation.box(extents=(20, 0.1, 20), transform=trans)
trans = np.eye(4)
trans[:3,3] = np.array([0, 10, 10])
frontwall = trimesh.creation.box(extents=(20, 20, 0.1), transform=trans)
trans[:3,3] = np.array([0, 10, -10])
backwall = trimesh.creation.box(extents=(20, 20, 0.1), transform=trans)
trans[:3,3] = np.array([-10, 10, 0])
leftwall = trimesh.creation.box(extents=(0.1, 20, 20), transform=trans)
trans[:3,3] = np.array([10, 10, 0])
rightwall = trimesh.creation.box(extents=(0.1, 20, 20), transform=trans)

combined = trimesh.util.concatenate(obj_meshes) # + [floor, frontwall, backwall, leftwall, rightwall])
scene = combined.scene()

#meshes_all = sum(obj_meshes)
#scene = meshes_all.scene()
scene.camera.resolution = [H, W]
[fov_x, fov_y] = get_fov_from_intrinsic(pix_T_cam)
scene.camera.fov = [fov_x, fov_y]
rot = np.array([[1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]])
world_T_camtri = np.dot(world_T_cam, rot)
scene.camera_transform = world_T_camtri

def pix_to_camrays(x,y, pix_T_cam, world_T_camtri):
    """
    x : [n]
    y : [n]
    pix_T_cam: 3x3
    """
    fx, fy, tx, ty = pix_T_cam[0,0], pix_T_cam[1,1], pix_T_cam[0,2], pix_T_cam[1,2]
    X = (x + 0.5 -tx)/fx
    Y = (-1) * (y + 0.5 -ty)/fy # make everything into trimesh coordiante
    vts = np.stack([X,Y,-np.ones_like(X)], axis=1).reshape([-1,3])

    # unitize
    vts = vts/np.linalg.norm(vts, axis=1)[..., np.newaxis]
    world_T_camtri_ = copy.deepcopy(world_T_camtri)
    world_T_camtri_[:3,3] = 0
    vts2 = apply_4x4(world_T_camtri_, vts)

    ori = np.ones_like(vts2) * world_T_camtri[:3,3]

    return ori, vts2

def pixel_to_depth_from_scene(combined, xv, yv, pix_T_cam, world_T_camtri):
    ori, vts2 = pix_to_camrays(xv, yv, pix_T_cam, world_T_camtri)
    points, index_ray, index_tri = combined.ray.intersects_location(
        ori, vts2, multiple_hits=False)
    # # for each hit, find the distance along its vector
    if points.shape[0] == 0:
        return np.array([]), np.array([])
    depth_ = trimesh.util.diagonal_dot(points - ori[0],
                                      vts2[index_ray])
    return depth_, index_ray

#np.dot(rot, cam_T_world)


scene.camera.z_near = camera_near_plane
scene.camera.z_far = camera_far_plane
#origins, vectors, pixels = scene.camera_rays()

# given pixels value, return depth
# 1. given pixels value, compute origins and vectors
xv, yv = np.meshgrid(np.linspace(0, H-1,H), np.linspace(0, W-1,W))
xv = xv.transpose().reshape([-1])
yv = yv.transpose().reshape([-1])


# depth_, index_ray = pixel_to_depth_from_scene(combined, xv, yv, pix_T_cam, world_T_camtri)

# #print("diff", np.sum(np.abs(vts2 - vectors)))

imageio.imsave("tmp/rendered.png", np.array(Image.open(io.BytesIO(scene.save_image()))))
imageio.imsave("tmp/rgb.png", videodata[timestep])
# depth_canvas = np.zeros((H*W))
# depth_canvas[index_ray] = depth_
# imageio.imsave("tmp/depth.png", depth_canvas.reshape([W,H]).transpose())
# import ipdb; ipdb.set_trace()


if use_obi:
    particle_positions = data['frames'][f"{timestep:04}"]['objects']['particles_positions']
    if timestep < data["static"]["start_frame_of_the_clip"]:
        obi_list = data['static0']["obi_object_ids"].tolist()
    else:
        obi_list = data['static']["obi_object_ids"].tolist()
    obj_pts = []
    for idx in obi_list:
        if str(idx) in particle_positions:
            pts = particle_positions[str(idx)]
            pts_cam = apply_4x4(cam_T_world, pts)
            pts_px = Camera2Pixels_np(pts_cam, pix_T_cam).astype(np.int32).clip(-1,256)
            selected_idx = np.where((pts_px[:,0] >= 0) * (pts_px[:,1] >=0) * (pts_px[:,0] <= 255) * (pts_px[:,1] <= 255))[0]

            pts_px = pts_px[selected_idx]
            pts_cam = pts_cam[selected_idx]
            #pts_px = pts_px[(pts_px[:,0] >= 0) * (pts_px[:,1] >=0), :]
            #pts_px = pts_px[(pts_px[:,0] <= 255) * (pts_px[:,1] <= 255), :]
            #534


            depth_, index_ray = pixel_to_depth_from_scene(combined, pts_px[:,0], pts_px[:,1], pix_T_cam, world_T_camtri)
            if index_ray.shape[0] > 0:
                index_ray = index_ray[pts_cam[index_ray, 2] > depth_]
            #print(pts_cam[index_ray, 2])
            #print(depth_)

            #pts = apply_4x4(cam_T_world, pts)
            #import ipdb; ipdb.set_trace()


            colors = np.zeros((pts.shape[0], 4), dtype=np.uint8)
            colors[:,3] = 255
            colors[:,2] = 255
            if index_ray.shape[0] > 0:
                colors[selected_idx[index_ray], 0] = 255
            obj_pts.append(trimesh.PointCloud(pts, colors=colors))





meshes = trimesh.Scene(obj_meshes_pts[0])
for i in range(1,len(obj_meshes_pts)):
   meshes += trimesh.Scene(obj_meshes_pts[i])
#(meshes + trimesh.Scene(obj_pts[0]) + camera + trimesh.Scene(floor) +  axis).show()




# colors = np.zeros((pts_cam.shape[0], 4), dtype=np.uint8)
# colors[:,3] = 255
# colors[:,0] = 255

# pcd = trimesh.PointCloud(pts_cam, colors=colors)
# (trimesh.Scene(pcd) + axis).show()


# 3. get segmentation input
#id_videodata = skvideo.io.vread(id_video_file)
import json
from pycocotools import mask as cocomask

seg_colors = data["static"]["video_object_segmentation_colors"]

obi_mask_from_proj = False
if use_obi:
    if data["static"]["obi_object_type"] == b"cloth":
        seg_colors = np.concatenate([seg_colors, data['static']['video_obi_object_segmentation_colors']], axis=0)
    else:
        obi_mask_from_proj = True

nobjs = seg_colors.shape[0]
n_obis = 0
if use_obi:
    n_obis = len(data['static']["obi_object_ids"].tolist())

with open(id_video_file, "r") as fh_read:
    in_dict = json.load(fh_read)

print("id_video_file", id_video_file)

# start_frame_id += 200

obj_info_dict = dict()
for timestep in range(start_frame_id):
    obj_info_dict[timestep] = dict()
    for obj_info in in_dict[f'{timestep:04}']:
        idx = obj_info["idx"]
        obj_info_dict[timestep][idx] = obj_info

import ipdb; ipdb.set_trace()

object_ids = data['static']["object_ids"].tolist()
red_id = object_ids.index(data["static"]["target_id"])
yellow_id = object_ids.index(data["static"]["zone_id"])

start_second_clip = data['static']["start_frame_of_the_clip"]
show_step = [x for x in range(10)] + [x for x in range(20, 30)] + [x for x in range(40, 50)]
show_step = show_step + [x + start_second_clip for x in show_step]


#for timestep in range(start_frame_id)
storage = np.zeros((start_frame_id, H,W))
for idx in range(nobjs + n_obis):
    frames = []
    for timestep in range(start_frame_id):
        if idx in obj_info_dict[timestep]:
            rgn = cocomask.decode(obj_info_dict[timestep][idx])
            frames.append((rgn * 255).astype(np.uint8))
        else:
            frames.append(np.zeros((H, W)))

        if red_id == idx and timestep in show_step:
            videodata[timestep][frames[-1]>10] = np.array([[255,0,0]])
            print("is_red", timestep)
            #print("hello")
            #videodata[timestep, :,:,:] =
        if yellow_id == idx and timestep in show_step:
            videodata[timestep][frames[-1]>10] = np.array([[255,255,0]])
            print("is_yellow", timestep)
    out = imageio.mimsave(
        os.path.join("tmp", f'id{idx}.gif'),
        frames, fps = 30)
    storage += np.stack(frames)/255.0
storage[storage > 0.5] = 1

import ipdb; ipdb.set_trace()
videodata[start_frame_id -50:] = 0
out = imageio.mimsave(
os.path.join("tmp", f"input_mask.gif"),
videodata, fps = 30)


if use_obi:
    from pycocotools import mask as cocomask
    from multiprocessing.pool import ThreadPool as Pool
    import multiprocessing

    import time
    start_time = time.time()
    import scipy
    sigma = 1.5
    thres = 5.0 #1.0
    n_obis = len(data['static0']["obi_object_ids"].tolist())

    if timestep < data["static"]["start_frame_of_the_clip"]:
        obi_list = data['static0']["obi_object_ids"].tolist()
    else:
        obi_list = data['static']["obi_object_ids"].tolist()
    obj_pts = []
    for obi_id in range(n_obis):
        #frames = []
        frames_before = []
        maxi = np.ones((3)) * float("-inf")
        mini = np.ones((3)) * float("inf")

        # start_frame_id


        def run_chunk(st, ed, queue):
            frames = []
            for timestep in range(st, ed + 1):

                #1. construct scene
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


                combined = trimesh.util.concatenate(obj_meshes) # + [floor, frontwall, backwall, leftwall, rightwall])
                scene = combined.scene()
                scene.camera.resolution = [H, W]
                scene.camera.fov = [fov_x, fov_y]
                scene.camera_transform = world_T_camtri
                scene.camera.z_near = camera_near_plane
                scene.camera.z_far = camera_far_plane

                #2. compute visible points
                particle_positions = data['frames'][f"{timestep:04}"]['objects']['particles_positions']
                if timestep < data["static"]["start_frame_of_the_clip"]:
                    idx = data['static0']["obi_object_ids"].tolist()[obi_id]
                else:
                    idx = data['static']["obi_object_ids"].tolist()[obi_id]

                canvas = np.zeros((H,W), np.uint8)
                #canvas_before = np.zeros((H,W))
                if str(idx) in particle_positions:
                    pts = particle_positions[str(idx)]
                    #if pts.shape[0] > 0:
                    #    maxi = np.maximum(maxi, np.max(pts, 0))
                    #    mini = np.minimum(mini, np.min(pts, 0))

                    pts_cam = apply_4x4(cam_T_world, pts)
                    pts_px = Camera2Pixels_np(pts_cam , pix_T_cam).astype(np.int32).clip(-1,256)
                    selected_idx = np.where((pts_px[:,0] >= 0) * (pts_px[:,1] >=0) * (pts_px[:,0] < W) * (pts_px[:,1] < H))[0]
                    pts_px = pts_px[selected_idx]
                    pts_cam = pts_cam[selected_idx]

                    # canvas_before[pts_px[:, 1], pts_px[:, 0]] = 255
                    # canvas_before = scipy.ndimage.filters.gaussian_filter(canvas_before, [sigma, sigma], mode='constant')
                    # canvas_before = canvas_before * (1 - storage[timestep])
                    # canvas_before[canvas_before > thres] = 255
                    #canvas[canvas_before > thres] = 125

                    if pts_px.shape[0] > 0:

                        #print("time", timestep, pts_px.shape)
                        canvas[pts_px[:, 1], pts_px[:, 0]] = 255

                        depth_, index_ray = pixel_to_depth_from_scene(combined, pts_px[:,0], pts_px[:,1], pix_T_cam, world_T_camtri)
                        if index_ray.shape[0] > 0:
                            index_ray = index_ray[pts_cam[index_ray, 2] > depth_] #occluded pixels
                            pts_px = pts_px[index_ray]
                            #print("remove pts, time", timestep, pts_px.shape)
                            canvas[pts_px[:, 1], pts_px[:, 0]] = 0 #remove occluded pixels
                    canvas = scipy.ndimage.filters.gaussian_filter(canvas, [sigma, sigma], mode='constant')
                    canvas[canvas > thres] = 255

                #if timestep == show_timestep:
                #    imageio.imsave(f"tmp/obi_id{obi_id}_showtime.png", canvas)
                    #import ipdb; ipdb.set_trace()

                n_nonzeros = np.sum(canvas > 125)
                if n_nonzeros == 0:
                    frames.append((timestep, 0))
                else:
                    bi_mask = np.asfortranarray(canvas > 125)
                    rels = cocomask.encode(bi_mask)
                    frames.append((timestep, rels))
                    #frames_before.append(canvas_before)
            #print("nframes", len(frames))
            queue.put(frames)
            return
            #return frames

        nchunks = 12
        st_eds = [(x[0], x[-1]) for x in np.array_split(range(start_frame_id), nchunks)]
        queues = [] #multiprocessing.Queue()
        processes = []
        for st_ed in st_eds:
            queue =  multiprocessing.Queue()
            p = multiprocessing.Process(target=run_chunk, args=(st_ed[0], st_ed[1], queue))
            processes.append(p)
            queues.append(queue)
            #p.start()

        for process in processes:
            process.start()
        #pool = Pool(nchunks)
        for process in processes:
            process.join()
        #for item in items:

        #result = pool.map(run_chunk, st_eds)
        result_list = []
        for queue in queues:
            while not queue.empty():
                result_list += queue.get()
        #pool.close()
        #pool.join()

        print("total time", time.time() - start_time)

        # decode coco
        frames = []
        for frame_id, rels in result_list:
            if isinstance(rels, int):
                frames.append(np.zeros((H, W), np.uint8))
            else:
                rels['counts'] = rels['counts'].decode('utf8')
                rgn = cocomask.decode(rels)
                frames.append((rgn * 255).astype(np.uint8))

        #print("maxi", maxi)
        #print("mini", mini)

        out = imageio.mimsave(
        os.path.join("tmp", f'obi_id{obi_id}.gif'),
        frames, fps = 30)

        # out = imageio.mimsave(
        # os.path.join("tmp", f'obi_id{obi_id}_before.gif'),
        # frames_before, fps = 30)


        writer = imageio.get_writer(f'tmp/obi_id{obi_id}.mp4', fps=20)
        for im_id, im in enumerate(frames):
            writer.append_data(np.concatenate([im, videodata[im_id, :,:,0]], axis=1))
        writer.close()

# for obi_ids, obi_pts in enumerate(obj_pts):
#     pts = obi_pts.vertices
#     pts_px = Camera2Pixels_np(apply_4x4(cam_T_world, pts), pix_T_cam).astype(np.int32).clip(0,255)
#     canvas = np.zeros((H,W))

#     canvas[pts_px[:, 0], pts_px[:, 1]] = 255
#     imageio.imsave("tmp/obi_mask.png", canvas)

#     import ipdb; ipdb.set_trace()


