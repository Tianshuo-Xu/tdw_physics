import random
from typing import List, Tuple, Dict, Optional
from abc import ABC
import h5py
import numpy as np
import pkg_resources
import io
import copy
import json
import scipy
import math
import trimesh
from scipy.spatial.transform import Rotation as R
from tdw.output_data import OutputData, Rigidbodies, Collision, EnvironmentCollision
from tdw.librarian import ModelRecord
from tdw.tdw_utils import TDWUtils
from tdw_physics.transforms_dataset import TransformsDataset
from tdw_physics.util import MODEL_LIBRARIES, str_to_xyz, xyz_to_arr, arr_to_xyz


def handle_random_transform_args(args):

    if args is not None:
        if type(args) in [float,int,bool]: # don't apply any parsing to simple datatypes
            return args
        try:
            args = json.loads(args)
        except:
            try: args = eval(args) #this allows us to read dictionaries etc.
            except: args = str_to_xyz(args)

        if hasattr(args, 'keys'):
            if 'class' in args:
                data = args['data']
                modname, classname = args['class']
                mod = importlib.import_module(modname)
                klass = get_attr(mod, classname)
                args = klass(data)
                assert callable(args)
            else:
                assert "x" in args, args
                assert "y" in args, args
                assert "z" in args, args
        elif hasattr(args, '__len__'):
            if len(args) == 3:
                args = {k:args[i] for i, k in enumerate(["x","y","z"])}
            else:
                assert len(args) == 2, (args, len(args))
        else:
            args + 0.0
    return args


class PhysicsInfo:
    """
    Physics info for an object.
    """

    def __init__(self,
                 record: ModelRecord,
                 mass: float,
                 dynamic_friction: float,
                 static_friction: float,
                 bounciness: float):
        """
        :param record: The model's metadata record.
        :param mass: The mass of the object.
        :param dynamic_friction: The dynamic friction.
        :param static_friction: The static friction.
        :param bounciness: The object's bounciness.
        """

        self.record = record
        self.mass = mass
        self.dynamic_friction = dynamic_friction
        self.static_friction = static_friction
        self.bounciness = bounciness


def _get_default_physics_info() -> Dict[str, PhysicsInfo]:
    """
    :return: The default object physics info from `data/physics_info.json`.
    """

    info: Dict[str, PhysicsInfo] = {}

    with io.open(pkg_resources.resource_filename(__name__, "data/physics_info.json"), "rt", encoding="utf-8") as f:
        _data = json.load(f)
        for key in _data:
            obj = _data[key]
            info[key] = PhysicsInfo(record=MODEL_LIBRARIES[obj["library"]].get_record(obj["name"]),
                                    mass=obj["mass"],
                                    bounciness=obj["bounciness"],
                                    dynamic_friction=obj["dynamic_friction"],
                                    static_friction=obj["static_friction"])
    return info


# The default physics info
PHYSICS_INFO: Dict[str, PhysicsInfo] = _get_default_physics_info()


def get_range(scl):
    if scl is None:
        return None, None
    elif hasattr(scl, '__len__'):
        return scl[0], scl[1]
    else:
        scl + 0.0
        return scl, scl


def get_random_xyz_transform(generator):
    if callable(generator):
        s = generator()
    elif hasattr(generator, 'keys'):
        sx0, sx1 = get_range(generator["x"])
        sx = random.uniform(sx0, sx1)
        sy0, sy1 = get_range(generator["y"])
        sy = random.uniform(sy0, sy1)
        sz0, sz1 = get_range(generator["z"])
        sz = random.uniform(sz0, sz1)
        s = {"x": sx, "y": sy, "z": sz}
    elif hasattr(generator, '__len__'):
        s0 = random.uniform(generator[0], generator[1])
        s = {"x": s0, "y": s0, "z": s0}
    else:
        generator + 0.0
        s = {"x": generator, "y": generator, "z": generator}
    assert hasattr(s, 'keys'), s
    assert 'x' in s, s
    assert 'y' in s, s
    assert 'z' in s, s
    return s


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



from pycocotools import mask as cocomask
class ObiSegsHelper():

    def __init__(self, image_size, camera_data):
        self.H, self.W = image_size
        # get camera extrinsic and intrinsic
        cam_T_world = np.reshape(camera_data['camera_matrix'], [4,4])
        world_T_cam = np.linalg.inv(cam_T_world)
        rot = np.array([[-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        world_T_cam  = np.dot(world_T_cam , rot)
        cam_T_world = np.linalg.inv(world_T_cam)

        proj_matrix = np.reshape(camera_data["projection_matrix"], [4,4])
        pix_T_cam = get_intrinsics_from_projection_matrix(proj_matrix, size=(self.H,self.W))
        [self.fov_x, self.fov_y] = get_fov_from_intrinsic(pix_T_cam)

        self.pix_T_cam = pix_T_cam
        self.cam_T_world = cam_T_world

        rot = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        self.world_T_camtri = np.dot(world_T_cam, rot)
        self.camera_far_plane=100
        self.camera_near_plane=0.1
        self.sigma = 1.5
        self.thres = 5.0 #1.0

    def get_obi_segs(self, data, timestep, n_obis):

        segs = []
        for obi_id in range(n_obis):
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
            scene.camera.resolution = [self.H, self.W]
            scene.camera.fov = [self.fov_x, self.fov_y]
            scene.camera_transform = self.world_T_camtri
            scene.camera.z_near = self.camera_near_plane
            scene.camera.z_far = self.camera_far_plane

            #2. compute visible points
            particle_positions = data['frames'][f"{timestep:04}"]['objects']['particles_positions']
            if "static0" in data and timestep < data["static"]["start_frame_of_the_clip"]:
                idx = data['static0']["obi_object_ids"].tolist()[obi_id]
            else:
                idx = data['static']["obi_object_ids"].tolist()[obi_id]

            canvas = np.zeros((self.H,self.W), np.uint8)
            #canvas_before = np.zeros((self.H,self.W))
            if str(idx) in particle_positions:
                pts = particle_positions[str(idx)]
                pts_cam = apply_4x4(self.cam_T_world, pts)
                pts_px = Camera2Pixels_np(pts_cam , self.pix_T_cam).astype(np.int32).clip(-1,256)
                selected_idx = np.where((pts_px[:,0] >= 0) * (pts_px[:,1] >=0) * (pts_px[:,0] < self.W) * (pts_px[:,1] < self.H))[0]
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

                    depth_, index_ray = pixel_to_depth_from_scene(combined, pts_px[:,0], pts_px[:,1], self.pix_T_cam, self.world_T_camtri)
                    if index_ray.shape[0] > 0:
                        index_ray = index_ray[pts_cam[index_ray, 2] > depth_] #occluded pixels
                        pts_px = pts_px[index_ray]
                        #print("remove pts, time", timestep, pts_px.shape)
                        canvas[pts_px[:, 1], pts_px[:, 0]] = 0 #remove occluded pixels
                canvas = scipy.ndimage.filters.gaussian_filter(canvas, [self.sigma, self.sigma], mode='constant')
                canvas[canvas > self.thres] = 255

            n_nonzeros = np.sum(canvas > 125)
            if n_nonzeros == 0:
              output =(timestep, 0)
            else:
              bi_mask = np.asfortranarray(canvas > 125)
              rels = cocomask.encode(bi_mask)
              output = (timestep, rels)

            segs.append(output)
        return segs


class RigidbodiesDataset(TransformsDataset, ABC):
    """
    A dataset for Rigidbody (PhysX) physics.
    """

    def __init__(self, port: int = 1071, monochrome: bool = False, **kwargs):

        TransformsDataset.__init__(self, port=port, **kwargs)

        self.physics_info: Dict[int, PhysicsInfo] = {}

        # Whether the objects will be set to the same color
        self.monochrome = monochrome


    def clear_static_data(self) -> None:
        super().clear_static_data()

        self.masses = np.empty(dtype=np.float32, shape=0)
        self.static_frictions = np.empty(dtype=np.float32, shape=0)
        self.dynamic_frictions = np.empty(dtype=np.float32, shape=0)
        self.bouncinesses = np.empty(dtype=np.float32, shape=0)
        self.colors = np.empty(dtype=np.float32, shape=(0,3))
        self.scales = []

    def _xyz_to_arr(self, xyz : dict):
        arr = np.array(
            [xyz[k] for k in ["x","y","z"]], dtype=np.float32)
        return arr

    def _arr_to_xyz(self, arr : np.ndarray):
        xyz = {k:arr[i] for i,k in enumerate(["x","y","z"])}
        return xyz

    def random_color(self, exclude=None, exclude_range=0.33):
        rgb = [random.random(), random.random(), random.random()]
        if exclude is None:
            return rgb

        assert len(exclude) == 3, exclude
        while any([np.abs(exclude[i] - rgb[i]) < exclude_range for i in range(3)]):
            rgb = [random.random(), random.random(), random.random()]

        return rgb



    def random_color_exclude_list(self, exclude_list=None, exclude_range=0.65, hsv_brightness=None):
        import colorsys
        if hsv_brightness is None:
            rgb = [random.random(), random.random(), random.random()]
        else:
            assert(hsv_brightness <= 1 and hsv_brightness>0), hsv_brightness
            rgb = [random.random(), random.random(), random.random()]
            scale = ((1-hsv_brightness) * random.random() + hsv_brightness)/np.max(rgb)
            rgb = [r*scale for r in rgb]
        if exclude_list is None or len(exclude_list) == 0:
            return rgb

        while True:
          bad_trial = False
          for exclude in exclude_list:

            assert len(exclude) == 3, exclude
            #if np.linalg.norm(exclude[i] - rgb[i]) < exclude_range:
            if np.linalg.norm(np.array([exclude[i] - rgb[i] for i in range(3)])) < exclude_range:
              bad_trial = True
              break
          if bad_trial:
            if hsv_brightness is None:
                rgb = [random.random(), random.random(), random.random()]
            else:
                rgb = [random.random(), random.random(), random.random()]
                scale = ((1-hsv_brightness) * random.random() + hsv_brightness)/np.max(rgb)
                rgb = [r*scale for r in rgb]

                #scale = ((1-hsv_brightness) * random.random() + hsv_brightness)/np.max(rgb)

                #hsv = (random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0)
                #rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])

            #  rgb = [random.random(), random.random(), random.random()]
          else:
              break
        return rgb

    def random_color_from_rng(self, exclude=None, exclude_range=0.33, seed=0):

        rng = np.random.RandomState(seed)
        rgb = [rng.random(), rng.random(), rng.random()]

        if exclude is None:
            return rgb
        assert len(exclude) == 3, exclude
        while any([np.abs(exclude[i] - rgb[i]) < exclude_range for i in range(3)]):
            rgb = [rng.random(), rng.random(), rng.random()]

        return rgb


    def get_random_scale_transform(self, scale):
        return get_random_xyz_transform(scale)

    def _add_name_scale_color(self, record, data) -> None:
        self.model_names.append(record.name)
        self.scales.append(data['scale'])
        self.colors = np.concatenate([self.colors, np.array(data['color']).reshape((1,3))], axis=0)

    def random_primitive(self,
                         object_types: List[ModelRecord],
                         scale: List[float] = [0.2, 0.3],
                         color: List[float] = None,
                         exclude_color: List[float] = None,
                         exclude_range: float = 0.25,
                         add_data: bool = True,
                         random_obj_id: bool = False
    ) -> dict:
        obj_record = random.choice(object_types)
        s = self.get_random_scale_transform(scale)

        obj_data = {
            "id": self.get_unique_id() if random_obj_id else self._get_next_object_id(),
            "scale": s,
            "color": np.array(color if color is not None else self.random_color(exclude_color, exclude_range)),
            "name": obj_record.name
        }

        if add_data:
            self._add_name_scale_color(obj_record, obj_data)
            # self.model_names.append(obj_data["name"])
            # self.scales.append(obj_data["scale"])
            # self.colors = np.concatenate([self.colors, obj_data["color"].reshape((1,3))], axis=0)
        return obj_record, obj_data

    def add_physics_object(self,
                           record: ModelRecord,
                           position: Dict[str, float],
                           rotation: Dict[str, float],
                           mass: float,
                           dynamic_friction: float,
                           static_friction: float,
                           bounciness: float,
                           o_id: Optional[int] = None,
                           add_data: Optional[bool] = True
    ) -> List[dict]:
        """
        Get commands to add an object and assign physics properties. Write the object's static info to the .hdf5 file.

        :param o_id: The unique ID of the object. If None, a random ID will be generated.
        :param record: The model record.
        :param position: The initial position of the object.
        :param rotation: The initial rotation of the object, in Euler angles.
        :param mass: The mass of the object.
        :param dynamic_friction: The dynamic friction of the object's physic material.
        :param static_friction: The static friction of the object's physic material.
        :param bounciness: The bounciness of the object's physic material.
        :param add_data: whether to add the chosen data to the hdf5

        :return: A list of commands: `[add_object, set_mass, set_physic_material]`
        """

        if o_id is None:
            o_id: int = self.get_unique_id()

        # Get the add_object command.
        add_object = self.add_transforms_object(o_id=o_id,
                                                record=record,
                                                position=position,
                                                rotation=rotation,
                                                add_data=add_data
                                                )

        if add_data:
            self.masses = np.append(self.masses, mass)
            self.dynamic_frictions = np.append(self.dynamic_frictions, dynamic_friction)
            self.static_frictions = np.append(self.static_frictions, static_friction)
            self.bouncinesses = np.append(self.bouncinesses, bounciness)

        # Log the physics info per object for easy reference in a controller.
        self.physics_info[o_id] = PhysicsInfo(record=record,
                                              mass=mass,
                                              dynamic_friction=dynamic_friction,
                                              static_friction=static_friction,
                                              bounciness=bounciness)

        # Return commands to create the object.
        return [add_object,
                {"$type": "set_mass",
                 "id": o_id,
                 "mass": mass},
                {"$type": "set_physic_material",
                 "id": o_id,
                 "dynamic_friction": dynamic_friction,
                 "static_friction": static_friction,
                 "bounciness": bounciness}]

    def add_physics_object_default(self,
                                   name: str,
                                   position: Dict[str, float],
                                   rotation: Dict[str, float],
                                   o_id: Optional[int] = None,
                                   add_data: Optional[bool] = True,
                                   ) -> List[dict]:
        """
        Add an object with default physics material values.

        :param o_id: The unique ID of the object. If None, a random ID number will be generated.
        :param name: The name of the model.
        :param position: The initial position of the object.
        :param rotation: The initial rotation of the object.

        :return: A list of commands: `[add_object, set_mass, set_physic_material]`
        """

        info = PHYSICS_INFO[name]
        return self.add_physics_object(o_id=o_id, record=info.record, position=position, rotation=rotation,
                                       mass=info.mass, dynamic_friction=info.dynamic_friction,
                                       static_friction=info.static_friction, bounciness=info.bounciness, add_data=add_data)

    def add_primitive(self,
                      record: ModelRecord,
                      position: Dict[str, float] = TDWUtils.VECTOR3_ZERO,
                      rotation: Dict[str, float] = TDWUtils.VECTOR3_ZERO,
                      scale: Dict[str, float] = {"x": 1., "y": 1., "z": 1},
                      o_id: Optional[int] = None,
                      material: Optional[str] = None,
                      color: Optional[list] = None,
                      exclude_color: Optional[list] = None,
                      mass: Optional[float] = 2.0,
                      dynamic_friction: Optional[float] = 0.1,
                      static_friction: Optional[float] = 0.1,
                      bounciness: Optional[float] = 0,
                      add_data: Optional[bool] = True,
                      scale_mass: Optional[bool] = True,
                      make_kinematic: Optional[bool] = False,
                      obj_list: Optional[list] = [],
                      ) -> List[dict]:

        cmds = []

        if o_id is None:
            o_id = self._get_next_object_id()

        if scale_mass:
            mass = mass * np.prod(xyz_to_arr(scale))

        # add the physics stuff
        cmds.extend(
            self.add_physics_object(
                record = record,
                position = position,
                rotation = rotation,
                mass = mass,
                dynamic_friction = dynamic_friction,
                static_friction = static_friction,
                bounciness = bounciness,
                o_id = o_id,
                add_data = add_data))

        # scale the object
        cmds.append(
            {"$type": "scale_object",
             "scale_factor": scale,
             "id": o_id})

        # set the material and color
        cmds.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(material)))

        color = color if color is not None else self.random_color(exclude=exclude_color)
        cmds.append(
            {"$type": "set_color",
             "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.},
             "id": o_id})

        if make_kinematic:
            cmds.extend([
                {"$type": "set_object_collision_detection_mode",
                 "mode": "continuous_speculative",
                 "id": o_id},
                {"$type": "set_kinematic_state",
                 "id": o_id,
                 "is_kinematic": True,
                 "use_gravity": True}])

        if add_data:
            data = {'name': record.name, 'id': o_id,
                    'scale': scale, 'color': color, 'material': material,
                    'mass': mass,
                    'dynamic_friction': dynamic_friction,
                    'static_friction': static_friction,
                    'bounciness': bounciness}
            self._add_name_scale_color(record, data)
            obj_list.append((record, data))

        return cmds


    def add_ramp(self,
                 record: ModelRecord = None,
                 position: Dict[str, float] = TDWUtils.VECTOR3_ZERO,
                 rotation: Dict[str, float] = TDWUtils.VECTOR3_ZERO,
                 scale: Dict[str, float] = {"x": 1., "y": 1., "z": 1},
                 o_id: Optional[int] = None,
                 material: Optional[str] = None,
                 color: Optional[list] = None,
                 mass: Optional[float] = None,
                 dynamic_friction: Optional[float] = None,
                 static_friction: Optional[float] = None,
                 bounciness: Optional[float] = None,
                 add_data: Optional[bool] = True
                 ) -> List[dict]:

        # get a named ramp or choose a random one
        ramp_records = {r.name: r for r in MODEL_LIBRARIES['models_full.json'].records \
                        if 'ramp' in r.name}
        if record is None: #record.name not in ramp_records.keys():
            record = ramp_records[random.choice(sorted(ramp_records.keys()))]

        cmds = []

        # add the ramp
        info = None
        if record.name in PHYSICS_INFO:
            info = PHYSICS_INFO[record.name]
        cmds.extend(
            self.add_physics_object(
                record = record,
                position = position,
                rotation = rotation,
                mass = mass or info.mass,
                dynamic_friction = dynamic_friction if dynamic_friction is not None else info.dynamic_friction,
                static_friction = static_friction if static_friction is not None else info.static_friction,
                bounciness = bounciness if bounciness is not None else info.bounciness,
                o_id = o_id,
                add_data = add_data))

        if o_id is None:
            o_id = cmds[-1]["id"]

        # scale the ramp
        cmds.append(
            {"$type": "scale_object",
             "scale_factor": scale,
             "id": o_id})

        # texture and color it
        cmds.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(material)))

        cmds.append(
            {"$type": "set_color",
             "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.},
             "id": o_id})

        # need to make ramp a kinetimatic object
        cmds.extend([
            {"$type": "set_object_collision_detection_mode",
             "mode": "continuous_speculative",
             "id": o_id},
            {"$type": "set_kinematic_state",
             "id": o_id,
             "is_kinematic": True,
             "use_gravity": True}])

        if add_data:
            self._add_name_scale_color(record, {'color': color, 'scale': scale})
            # self.model_names.append(record.name)
            # self.colors = np.concatenate([self.colors, np.array(color).reshape((1,3))], axis=0)
            # self.scales.append(scale)

        return cmds

    def get_objects_by_mass(self, mass: float) -> List[int]:
        """
        :param mass: The mass threshold.

        :return: A list of object IDs for objects with mass <= the mass threshold.
        """

        return [o for o in self.physics_info.keys() if self.physics_info[o].mass < mass]

    def get_falling_commands(self, mass: float = 3) -> List[List[dict]]:
        """
        :param mass: Objects with <= this mass might receive a force.

        :return: A list of lists; per-frame commands to make small objects fly up.
        """

        per_frame_commands: List[List[dict]] = []

        # Get a list of all small objects.
        small_ids = self.get_objects_by_mass(mass)
        random.shuffle(small_ids)
        max_num_objects = len(small_ids) if len(small_ids) < 8 else 8
        min_num_objects = max_num_objects - 3
        if min_num_objects <= 0:
            min_num_objects = 1
        # Add some objects.
        for i in range(random.randint(min_num_objects, max_num_objects)):
            o_id = small_ids.pop(0)
            force_dir = np.array([random.uniform(-0.125, 0.125), random.uniform(0.7, 1), random.uniform(-0.125, 0.125)])
            force_dir = force_dir / np.linalg.norm(force_dir)
            min_force = self.physics_info[o_id].mass * 2
            max_force = self.physics_info[o_id].mass * 4
            force = TDWUtils.array_to_vector3(force_dir * random.uniform(min_force, max_force))
            per_frame_commands.append([{"$type": "apply_force_to_object",
                                        "force": force,
                                        "id": o_id}])
            # Wait some frames.
            for j in range(10, 30):
                per_frame_commands.append([])
        return per_frame_commands

    def _get_send_data_commands(self) -> List[dict]:
        commands = super()._get_send_data_commands()
        commands.extend([{"$type": "send_collisions",
                          "enter": True,
                          "exit": True,
                          "stay": True,
                          "collision_types": ["obj", "env"]},
                         {"$type": "send_rigidbodies",
                          "frequency": "always"}])

        if self.save_meshes:
            commands.append({"$type": "send_meshes", "frequency": "once"})

        return commands

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        ## physical
        static_group.create_dataset("mass", data=self.masses)
        static_group.create_dataset("static_friction", data=self.static_frictions)
        static_group.create_dataset("dynamic_friction", data=self.dynamic_frictions)
        static_group.create_dataset("bounciness", data=self.bouncinesses)

        ## size and colors
        static_group.create_dataset("color", data=self.colors)
        if len(self.scales) > 0:
          static_group.create_dataset("scale", data=np.stack([xyz_to_arr(_s) for _s in self.scales], 0))
          static_group.create_dataset("scale_x", data=[_s["x"] for _s in self.scales])
          static_group.create_dataset("scale_y", data=[_s["y"] for _s in self.scales])
          static_group.create_dataset("scale_z", data=[_s["z"] for _s in self.scales])

        static_group.create_dataset("use_obi", data=self.use_obi)
        #if self.use_obi:
        #    static_group.create_dataset("obi_object_ids", data=self.obi.actors)

        if self.save_meshes:
            mesh_group = static_group.create_group("mesh")

            obj_points = []
            for idx, object_id in enumerate(self.object_ids):
                vertices, faces = self.object_meshes[object_id]
                mesh_group.create_dataset(f"faces_{idx}", data=faces)
                mesh_group.create_dataset(f"vertices_{idx}", data=vertices)

    def _write_frame(self, frames_grp: h5py.Group, resp: List[bytes], frame_num: int) -> \
            Tuple[h5py.Group, h5py.Group, dict, bool]:
        frame, objs, tr, done = super()._write_frame(frames_grp=frames_grp, resp=resp, frame_num=frame_num)
        num_objects = len(self.object_ids)
        # Physics data.
        velocities = np.empty(dtype=np.float32, shape=(num_objects, 3))
        angular_velocities = np.empty(dtype=np.float32, shape=(num_objects, 3))
        # Collision data.
        collision_ids = np.empty(dtype=np.int32, shape=(0, 2))
        collision_relative_velocities = np.empty(dtype=np.float32, shape=(0, 3))
        collision_contacts = np.empty(dtype=np.float32, shape=(0, 2, 3))
        collision_states = np.empty(dtype=str, shape=(0, 1))
        # Environment Collision data.
        env_collision_ids = np.empty(dtype=np.int32, shape=(0, 1))
        env_collision_contacts = np.empty(dtype=np.float32, shape=(0, 2, 3))

        sleeping = True

        # rtypes = [OutputData.get_data_type_id(r) for r in resp[:-1]]
        # print(frame_num, "COLLISION" if 'coll' in rtypes else "")

        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "rigi":
                ri = Rigidbodies(r)
                ri_dict = dict()
                for i in range(ri.get_num()):
                    ri_dict.update({ri.get_id(i): {"vel": ri.get_velocity(i),
                                                   "ang": ri.get_angular_velocity(i)}})
                    # Check if any objects are sleeping that aren't in the abyss.
                    if not ri.get_sleeping(i) and tr[ri.get_id(i)]["pos"][1] >= -1:
                        sleeping = False
                # Add the Rigibodies data.
                for o_id, i in zip(self.object_ids, range(num_objects)):
                    try:
                        velocities[i] = ri_dict[o_id]["vel"]
                        angular_velocities[i] = ri_dict[o_id]["ang"]
                    except KeyError:
                        print("Couldn't store velocity data for object %d" % o_id)
                        print("frame num", frame_num)
                        print("ri_dict", ri_dict)
                        print([OutputData.get_data_type_id(r) for r in resp])
            elif r_id == "coll":
                co = Collision(r)
                collision_states = np.append(collision_states, co.get_state())
                collision_ids = np.append(collision_ids, [co.get_collider_id(), co.get_collidee_id()])
                collision_relative_velocities = np.append(collision_relative_velocities, co.get_relative_velocity())
                for i in range(co.get_num_contacts()):
                    collision_contacts = np.append(collision_contacts, (co.get_contact_normal(i),
                                                                        co.get_contact_point(i)))
            elif r_id == "enco":
                en = EnvironmentCollision(r)
                env_collision_ids = np.append(env_collision_ids, en.get_object_id())
                for i in range(en.get_num_contacts()):
                    env_collision_contacts = np.append(env_collision_contacts, (en.get_contact_normal(i),
                                                                                en.get_contact_point(i)))
        objs.create_dataset("velocities", data=velocities.reshape(num_objects, 3), compression="gzip")
        objs.create_dataset("angular_velocities", data=angular_velocities.reshape(num_objects, 3), compression="gzip")
        collisions = frame.create_group("collisions")
        collisions.create_dataset("object_ids", data=collision_ids.reshape((-1, 2)), compression="gzip")
        collisions.create_dataset("relative_velocities", data=collision_relative_velocities.reshape((-1, 3)),
                                  compression="gzip")
        collisions.create_dataset("contacts", data=collision_contacts.reshape((-1, 2, 3)), compression="gzip")
        collisions.create_dataset("states", data=collision_states.astype('S'), compression="gzip")
        env_collisions = frame.create_group("env_collisions")
        env_collisions.create_dataset("object_ids", data=env_collision_ids, compression="gzip")
        env_collisions.create_dataset("contacts", data=env_collision_contacts.reshape((-1, 2, 3)),
                                      compression="gzip")
        return frame, objs, tr, sleeping

    def get_object_target_collision(self, obj_id: int, target_id: int, resp: List[bytes]):

        assert(target_id in self.object_ids.tolist())
        assert(obj_id in self.object_ids.tolist())
        contact_points = []
        contact_normals = []

        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "coll":
                co = Collision(r)
                coll_ids = [co.get_collider_id(), co.get_collidee_id()]
                if [obj_id, target_id] == coll_ids or [target_id, obj_id] == coll_ids:
                    contact_points = [co.get_contact_point(i) for i in range(co.get_num_contacts())]
                    contact_normals = [co.get_contact_normal(i) for i in range(co.get_num_contacts())]
        return (contact_points, contact_normals)

    def get_object_environment_collision(self, obj_id: int, resp: List[bytes]):

        contact_points = []
        contact_normals = []

        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == 'enco':
                en = EnvironmentCollision(r)
                if en.get_object_id() == obj_id:
                    contact_points = [np.array(en.get_contact_point(i)) for i in range(en.get_num_contacts())]
                    contact_normals = [np.array(en.get_contact_normal(i)) for i in range(en.get_num_contacts())]

        return (contact_points, contact_normals)
