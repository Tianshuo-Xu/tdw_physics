import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abc import ABC
import numpy as np
from tdw.librarian import ModelRecord
from rigidbodies_dataset import RigidbodiesDataset, PHYSICS_INFO, get_random_xyz_transform
from dataset import MATERIAL_NAMES, MATERIAL_TYPES
from tdw.output_data import OutputData, Collision, EnvironmentCollision
from dataclasses import dataclass
from typing import List, Dict, Optional
from scipy.stats import vonmises, norm, lognorm
from noisy_utils import *
import json
from tdw.tdw_utils import TDWUtils
from util import xyz_to_arr, MODEL_LIBRARIES

XYZ = ['x', 'y', 'z']


@dataclass
class RigidNoiseParams:
    """
    TODO: confirm collision noise on force vs. momentum

    Holders for parameters defining the amount of noise in noisy simulations

    All noise parameters can take a `None` value to remove noise for that particular parameter; `None` is the default for all noise values

    position: Dict[str, float]: Gaussian noise around position of all dynamic objects (separate noise for each dimension)

    rotation: Dict[str, float]: vonMises precision for rotation noise along the x,y,z axes (separate noise for each dimension)

    velocity_dir: Dict[str, float]: vonMises precision for noise in the initial velocity direction along the x,y,z axes (separate noise for each dimension)

    velocity_mag: float: log-normal noise around the magnitude of initial speed

    mass: float: log-normal noise around the true object mass

    static_friction: float: Gaussian noise around the true object static friction

    dynamic_friction: float: Gaussian noise around the true object dynamic friction

    bounciness: float: Gaussian noise around the true object bounciness

    collision_dir: Dict[str, float]: vonMises precision for noise injected into the force normals for each collision along the x,y,z axes (separate noise for each dimension)

    collision_mag: float: log-normal noise around the true magnitude of the force normals for a collision

    coll_threshold: float: collisions below this threshold are ignored when adding noise
    """

    position: Dict[str, float] = None
    rotation: Dict[str, float] = None
    velocity_dir: Dict[str, float] = None
    velocity_mag: float = None
    mass: float = None
    static_friction: float = None
    dynamic_friction: float = None
    bounciness: float = None
    collision_dir: float = None
    collision_mag: float = None
    coll_threshold: float = None

    def save(self, flpth):
        selfobj = {
            'position': self.position,
            'rotation': self.rotation,
            'velocity_dir': self.velocity_dir,
            'velocity_mag': self.velocity_mag,
            'mass': self.mass,
            'static_friction': self.static_friction,
            'dynamic_friction': self.dynamic_friction,
            'bounciness': self.bounciness,
            'collision_dir': self.collision_dir,
            'collision_mag': self.collision_mag,
            'coll_threshold': self.coll_threshold
        }
        with open(flpth, 'w') as ofl:
            json.dump(selfobj, ofl)

    # def to_string(self):
    #     return'-'.join([x for x in .values()])

    @staticmethod
    def load(flpth):
        with open(flpth, 'r') as ifl:
            obj = json.load(ifl)
        return RigidNoiseParams(**obj)


class IsotropicRigidNoiseParams(RigidNoiseParams):
    """
    As RigidNoiseParams except all [x,y,z] noises become floats
    """

    def __init__(self,
                 position: float = None,
                 rotation: float = None,
                 velocity_dir: float = None,
                 collision_dir: float = None,
                 **kwargs):
        if position is not None:
            position = dict([[k, position] for k in XYZ])
        if rotation is not None:
            rotation = dict([[k, rotation] for k in XYZ])
        if velocity_dir is not None:
            velocity_dir = dict([[k, velocity_dir] for k in XYZ])
        # if collision_dir is not None:
        #     collision_dir = dict([[k, collision_dir] for k in XYZ])
        RigidNoiseParams.__init__(self, position,
                                  rotation, velocity_dir,
                                  collision_dir, **kwargs)


NO_NOISE = RigidNoiseParams()



class NoisyRigidbodiesDataset(RigidbodiesDataset, ABC):
    """
    A dataset for running Rigidbody (PhysX) physics with injected noise
    """

    def __init__(self,
                 noise: RigidNoiseParams = NO_NOISE,
                 read_global_seed = None,
                 read_trial_seed = None,
                 **kwargs):
        RigidbodiesDataset.__init__(self, **kwargs)
        self._noise_params = noise

        # how to generate collision noise
        # self._ongoing_collisions = []
        # self._lasttime_collisions = []
        self.set_collision_noise_generator(noise)
        # if self.collision_noise_generator is not None:
        #     print("example noise", self.collision_noise_generator())
        self._registered_objects = []

        self.read_trial_seed = read_trial_seed  if read_trial_seed else None
        self.read_global_seed = read_global_seed  if read_global_seed else None

    @staticmethod
    def get_random_avatar_position(trial_level_random,
                                   radius_min: float,
                                   radius_max: float,
                                   y_min: float,
                                   y_max: float,
                                   center: Dict[str, float],
                                   angle_min: float = 0,
                                   angle_max: float = 360,
                                   reflections: bool = False,
                                   ) -> Dict[str, float]:
        """
        Overwrites method from rigidbodies_dataset to fix angle
        """

        a_r = trial_level_random.uniform(radius_min, radius_max)
        a_x = center["x"] + a_r
        a_z = center["z"] + a_r
        theta = np.radians(trial_level_random.uniform(angle_min, angle_max))
        if reflections:
            theta2 = trial_level_random.uniform(angle_min+180, angle_max+180)
            theta = trial_level_random.choice([theta, theta2])
        a_y = trial_level_random.uniform(y_min, y_max) + center["y"]
        a_x_new = np.cos(theta) * (a_x - center["x"]) - np.sin(theta) * (a_z - center["z"]) + center["x"]
        a_z_new = np.sin(theta) * (a_x - center["x"]) + np.cos(theta) * (a_z - center["z"]) + center["z"]
        a_x = a_x_new
        a_z = a_z_new

        return {"x": a_x, "y": a_y, "z": a_z}

    def random_model(self,
                     trial_level_random,
                     object_types: List[ModelRecord],
                     random_obj_id: bool = False,
                     add_data: bool = True) -> dict:
        obj_record = trial_level_random.choice(object_types)
        obj_data = {
            "id": self.get_unique_id() if random_obj_id else self._get_next_object_id(),
            "name": obj_record.name
        }

        if add_data:
            self.model_names.append(obj_data["name"])

        return obj_record, obj_data

    def get_material_name(self, trial_level_random, material):

        if material is not None:
            if material in MATERIAL_TYPES:
                mat = trial_level_random.choice(MATERIAL_NAMES[material])
            else:
                assert any((material in MATERIAL_NAMES[mtype] for mtype in self.material_types)), \
                    (material, self.material_types)
                mat = material
        else:
            mtype = trial_level_random.choice(self.material_types)
            mat = trial_level_random.choice(MATERIAL_NAMES[mtype])
        return mat

    def random_color(self, trial_level_random, exclude=None, exclude_range=0.33):
        rgb = [trial_level_random.random(), trial_level_random.random(), trial_level_random.random()]
        if exclude is None:
            return rgb

        assert len(exclude) == 3, exclude
        while any([np.abs(exclude[i] - rgb[i]) < exclude_range for i in range(3)]):
            rgb = [trial_level_random.random(), trial_level_random.random(), trial_level_random.random()]

        return rgb

    def add_ramp(self,
                 trial_level_random,
                 record: ModelRecord,
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
        ramp_records = {r.name: r for r in MODEL_LIBRARIES['models_full.json'].records
                        if 'ramp' in r.name}
        if record.name not in ramp_records.keys():
            record = ramp_records[trial_level_random.choice(sorted(ramp_records.keys()))]

        cmds = []

        # add the ramp
        info = PHYSICS_INFO[record.name]
        cmds.extend(
            self.add_physics_object(
                record=record,
                position=position,
                rotation=rotation,
                mass=mass or info.mass,
                dynamic_friction=dynamic_friction or info.dynamic_friction,
                static_friction=static_friction or info.static_friction,
                bounciness=bounciness or info.bounciness,
                o_id=o_id,
                add_data=add_data))

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
                record, o_id, self.get_material_name(trial_level_random, material)))

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
            self._add_name_scale_color(
                record, {'color': color, 'scale': scale})
            # self.model_names.append(record.name)
            # self.colors = np.concatenate([self.colors, np.array(color).reshape((1,3))], axis=0)
            # self.scales.append(scale)

        return cmds

    def get_random_scale_transform(self, scale, trial_level_random):
        return get_random_xyz_transform(scale, trial_level_random)

    def random_primitive(self,
                         trial_level_random,
                         object_types: List[ModelRecord],
                         scale: List[float] = [0.2, 0.3],
                         color: List[float] = None,
                         exclude_color: List[float] = None,
                         exclude_range: float = 0.25,
                         add_data: bool = True,
                         random_obj_id: bool = False
                         ) -> dict:
        obj_record = trial_level_random.choice(object_types)
        s = self.get_random_scale_transform(scale, trial_level_random)

        obj_data = {
            "id": self.get_unique_id() if random_obj_id else self._get_next_object_id(),
            "scale": s,
            "color": np.array(color if color is not None else self.random_color(trial_level_random, exclude_color, exclude_range)),
            "name": obj_record.name
        }

        if add_data:
            self._add_name_scale_color(obj_record, obj_data)
            # self.model_names.append(obj_data["name"])
            # self.scales.append(obj_data["scale"])
            # self.colors = np.concatenate([self.colors, obj_data["color"].reshape((1,3))], axis=0)
        return obj_record, obj_data


    def add_primitive(self,
                      trial_level_random,
                      sub_level_seed: int,
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
                      apply_texture: Optional[bool] = True
                      ) -> List[dict]:
        """
        Overwrites method from rigidbodies_dataset to add noise to objects when added to the scene
        """
        cmds = []

        if o_id is None:
            o_id = self._get_next_object_id()

        if scale_mass:
            mass = mass * np.prod(xyz_to_arr(scale))

        # add the physics stuff
        cmds.extend(
            self.add_physics_object(
                sub_level_seed=sub_level_seed,
                record=record,
                position=position,
                rotation=rotation,
                mass=mass,
                dynamic_friction=dynamic_friction,
                static_friction=static_friction,
                bounciness=bounciness,
                o_id=o_id,
                add_data=add_data))

        # scale the object
        cmds.append(
            {"$type": "scale_object",
             "scale_factor": scale,
             "id": o_id})

        # set the material and color
        if apply_texture:
            cmds.extend(
                self.get_object_material_commands(
                    record, o_id, self.get_material_name(trial_level_random, material)))

            color = color if color is not None else self.random_color(trial_level_random,
                exclude=exclude_color)
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

    def add_physics_object(self,
                           sub_level_seed: int,
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
        Overwrites method from rigidbodies_dataset to add noise to objects when added to the scene
        """
        # print("----------------------------------------------------------------------------------------------------------------------------")
        # print("original o_id: ", o_id)
        # # print("noisy_params: ", self._noise_params)
        # print("original positions: ", position)
        # print("original rotations: ", rotation)
        # print("original masses: ", mass)
        # print("original dynamic_frictions: ", dynamic_friction)
        # print("original static_frictions: ", static_friction)
        # print("original bouncinesses: ", bounciness)
        n = self._noise_params
        rotrad = dict([[k, deg2rad(rotation[k])]
                       for k in rotation.keys()])
        for k in XYZ:
            if n.position is not None and k in n.position.keys()\
                    and n.position[k] is not None:
                position[k] = norm.rvs(position[k],
                                       n.position[k], random_state=sub_level_seed)
            # this is adding vonmises noise to the Euler angles
            if n.rotation is not None and k in n.rotation.keys()\
                    and n.rotation[k] is not None:
                rotrad[k] = vonmises.rvs(n.rotation[k], rotrad[k], random_state=sub_level_seed)
        rotation = dict([[k, rad2deg(rotrad[k])]
                         for k in rotrad.keys()])
        if n.mass is not None:
            mass *= lognorm.rvs(n.mass, loc=0, random_state=sub_level_seed)
        # Clamp frictions to be > 0
        if n.dynamic_friction is not None:
            dynamic_friction = max(0,
                                   norm.rvs(dynamic_friction,
                                            n.dynamic_friction, random_state=sub_level_seed))
        if n.static_friction is not None:
            static_friction = max(0,
                                  norm.rvs(static_friction,
                                           n.static_friction, random_state=sub_level_seed))
        # Clamp bounciness between 0 and 1
        if n.bounciness is not None:
            bounciness = max(0, min(1,
                                    norm.rvs(bounciness,
                                             n.bounciness, random_state=sub_level_seed)))
        # print("perturbed positions: ", position)
        # print("perturbed rotations: ", rotation)
        # print("perturbed masses: ", mass)
        # print("perturbed dynamic_frictions: ", dynamic_friction)
        # print("perturbed static_frictions: ", static_friction)
        # print("perturbed bouncinesses: ", bounciness)
        # print("----------------------------------------------------------------------------------------------------------------------------")
        self._registered_objects.append(o_id)
        return RigidbodiesDataset.add_physics_object(
            self,
            record, position, rotation, mass,
            dynamic_friction, static_friction,
            bounciness, o_id, add_data
        )

    def get_per_frame_commands(self, resp: List[bytes], frame: int, sub_level_seed: int) -> List[dict]:
        """
        Overwrites abstract method to add collision noise commands

        Inhereting classes should *always* extend this output

        self._ongoing_collisions = []
        self._lasttime_collisions = []
        """
        cmds = []
        if self.collision_noise_generator is not None:
            coll_data = self._get_collision_data(resp)
            if len(coll_data) > 0:
                # """ some filtering and smoothing can be done below, e.g remove target zone collision
                for cd in coll_data:
                    # if '1' not in nm_ap:
                    # print("----------------------------------------------------------------------------------------------------------------------------")
                    coll_noise_cmds = self.apply_collision_noise(sub_level_seed, cd)
                    cmds.extend(coll_noise_cmds)

                # new_collisions = []
                # for cd in coll_data:
                #     nm_ap = str(cd['agent_id']) + '_' + str(cd['patient_id'])
                #     if '3' in nm_ap:
                #         # Ensure consistency in naming / comparisons
                #         aid = cd['agent_id']
                #         pid = cd['patient_id']
                #         # nm_ap = str(aid) + '_' + str(pid)

                #         for r in resp[:-1]:
                #             r_id = OutputData.get_data_type_id(r)
                #             if r_id == "rigi":
                #                 ri = Rigidbodies(r)
                #                 for i in range(ri.get_num()):
                #                     if ri.get_id(i) == aid:
                #                         va = ri.get_velocity(i)
                #                     elif ri.get_id(i) == pid:
                #                         vp = ri.get_velocity(i)
                #         vel_diff = np.subtract(va, vp)
                #         if np.any(cd['impulse']):
                #             print('agent: ', aid)
                #             print('patient: ', pid)
                #             print('corr impulse delvel: ', np.dot(cd['impulse'], vel_diff)/(np.linalg.norm(cd['impulse'])*np.linalg.norm(vel_diff)))
                    # if cd['state'] == 'enter':
                    #     print('start rvel: ' + nm_ap + ' : '
                    #           + str(va) + '; ' + str(vp))
                    #     self._ongoing_collisions.append(nm_ap)
                    #     new_collisions.append(nm_ap)
                    # else:
                    #     if nm_ap in self._lasttime_collisions:
                    #         print('next rvel: ' + nm_ap + ' : '
                    #               + str(va) + '; ' + str(vp))

                    # if cd['state'] == 'exit':
                    #     self._ongoing_collisions = [
                    #         c for c in self._ongoing_collisions if
                    #         c != nm_ap
                    #     ]
                # self._lasttime_collisions = new_collisions
                # """
                # coll_noise_cmds = self.apply_collision_noise(coll_data)
                # cmds.extend(coll_noise_cmds)

        return cmds

    def apply_collision_noise(self, sub_level_seed: int,  data=None):
        nm_ap = str(data['agent_id']) + '_' + str(data['patient_id'])
        # print('collision objects: ', nm_ap)
        if data is None or np.linalg.norm(data['impulse']) < self._noise_params.coll_threshold:
            # print("NOT PERTURBED")
            return []
        # print("YES PERTURBED")
        # print("num_contacts: ", data['num_contacts'])
        p_id = data['patient_id']
        a_id = data['agent_id']
        # print('contact points: ', data['contact_points'])
        contact_points = [dict([[k, pt[idx]] for idx, k in enumerate(XYZ)]) for pt in data['contact_points']]
        # print('contact points formatted: ', contact_points)
        impulse = dict([[k, data['impulse'][idx]] for idx, k in enumerate(XYZ)])
        # print("original impulse: ", impulse)
        # print("norm original impulse: ", np.linalg.norm(list(impulse.values())))
        force = self.collision_noise_generator(sub_level_seed, data['impulse'])
        delta_force = self._calculate_collision_differential(impulse, force)
        # print("perturbed impulse: ", force)
        # print("norm perturbed impulse: ", np.linalg.norm(list(force.values())))
        # print("perturbed impulse delta: ", delta_force)
        force_avg_p = dict([[k, delta_force[k]/data['num_contacts']] for k in XYZ])
        force_avg_a = dict([[k, -delta_force[k]/data['num_contacts']] for k in XYZ])
        # if data['num_contacts'] != 0:
        #     force_avg_p = dict([[k, delta_force[k]/data['num_contacts']] for k in XYZ])
        #     force_avg_a = dict([[k, -delta_force[k]/data['num_contacts']] for k in XYZ])
        # else:
        #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     force_avg_p = delta_force
        #     force_avg_a = dict([[k, -delta_force[k]] for k in XYZ])
        # print("perturbed force_avg_p: ", force_avg_p)
        # print("perturbed force_avg_a: ", force_avg_a)        
        # see here: https://github.com/threedworld-mit/tdw/blob/ce177b9754e4fa7bc7094c59937bb12c01f978aa/Documentation/api/command_api.md#apply_force_at_position
        # why there are multiple contact normals: https://gamedev.stackexchange.com/questions/40048/why-doesnt-unitys-oncollisionenter-give-me-surface-normals-and-whats-the-mos
        cmds_p = [
            {
                "$type": "apply_force_at_position",
                "force": force_avg_p,
                "position": contact_point,
                "id": p_id
            } for contact_point in contact_points
        ]
        cmds_a = [
            {
                "$type": "apply_force_at_position",
                "force": force_avg_a,
                "position": contact_point,
                "id": a_id
            } for contact_point in contact_points
        ]
        cmds = cmds_p+cmds_a
        # print('cmd: ', cmds)
        # print("----------------------------------------------------------------------------------------------------------------------------")
        return cmds

    # """ INCOMPLETE - Adds force but not relative """

    """
    Helper function that takes in a collision momentum transfer vector, then calculates the momentum to apply
    in the next timestep to actualize the collision noise
    """
    def _calculate_collision_differential(self, original_momentum: Dict[str, float], perturbed_momentum: Dict[str, float]) -> Dict[str, float]:
        return dict([[k, perturbed_momentum[k]-original_momentum[k]] for k in XYZ])

    def set_collision_noise_generator(self,
                                      noise_obj: RigidNoiseParams):
        # Only make noise if there is noise to be added
        # if noise_obj.collision_dir is not None:
        #     ncd = copy.copy(noise_obj.collision_dir)
        # else:
        #     ncd = None
        ncd = noise_obj.collision_dir
        ncm = noise_obj.collision_mag
        if ncd is None and ncm is None:
            self.collision_noise_generator = None
        else:
            """ NOTE MAKE THIS ALL RELATIVE | ADD INPUT """
            if ncm is None:
                def cng(sub_level_seed, impulse):
                    impulse_rand_dir = rand_von_mises_fisher(sub_level_seed, impulse/np.linalg.norm(impulse),kappa=ncd)[0]
                    return rotmag2vec(dict([[k, impulse_rand_dir[idx]]
                                            for idx, k in enumerate(XYZ)]),
                                      np.linalg.norm(impulse))
            elif ncd is None:
                def cng(sub_level_seed, impulse):
                    impulse_dir = impulse/np.linalg.norm(impulse)
                    return rotmag2vec(dict([[k, impulse_dir[idx]] for idx, k in enumerate(XYZ)]),
                                      np.linalg.norm(impulse))*lognorm.rvs(ncm, loc=0, random_state=sub_level_seed)
            else:
                def cng(sub_level_seed, impulse):
                    impulse_rand_dir = rand_von_mises_fisher(sub_level_seed, impulse/np.linalg.norm(impulse),kappa=ncd)[0]
                    return rotmag2vec(dict([[k, impulse_rand_dir[idx]]
                                            for idx, k in enumerate(XYZ)]),
                                      np.linalg.norm(impulse)*lognorm.rvs(ncm, loc=0, random_state=sub_level_seed))
            self.collision_noise_generator = cng

    def settle(self):
        """
        After adding a set of objects in a noisy way,
        'settles' the world to avoid interpenetration
        check out this: https://github.com/threedworld-mit/tdw/blob/ce177b9754e4fa7bc7094c59937bb12c01f978aa/Documentation/lessons/semantic_states/overlap.md
        """
        # print("applying settle function!")
        cmds = []
        # disable gravity for all added objects
        for o_id in self._registered_objects:
            # print("disabling gravity for object: ", o_id)
            cmds.extend([{"$type": "set_kinematic_state",
                            "id": o_id,
                            "use_gravity": False}])
        
        # slowly resolve possible collisions
        cmds.extend([{"$type": "set_time_step",
                            "time_step": 0.0001},
                    {"$type": "step_physics",
                            "frames": 500}])

        # enable gravity again
        for o_id in self._registered_objects:
            # print("enabling object: ", o_id)
            cmds.extend([{"$type": "set_kinematic_state",
                            "id": o_id,
                            "use_gravity": True}])

        # set physics speed to normal
        cmds.extend([{"$type": "set_time_step",
                                "time_step": 0.03}])
        self._registered_objects = []
        # print("finished applying settle function!")

        return cmds

    """ Ensures collision data is sent pre (change for post) """

    def _get_send_data_commands(self) -> List[dict]:
        commands = super()._get_send_data_commands()
        # Can't send this more than once...
        commands = [c for c in commands if c['$type'] != 'send_collisions']
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

    def _get_collision_data(self, resp: List[bytes]):
        coll_data = []
        r_ids = [OutputData.get_data_type_id(r) for r in resp[:-1]]
        for i, r_id in enumerate(r_ids):
            if r_id == 'coll':
                co = Collision(resp[i])
                state = co.get_state()
                agent_id = co.get_collider_id()
                patient_id = co.get_collidee_id()
                relative_velocity = co.get_relative_velocity()
                impulse = co.get_impulse()
                num_contacts = co.get_num_contacts()
                contact_points = [co.get_contact_point(i)
                                  for i in range(num_contacts)]
                contact_normals = [co.get_contact_normal(i)
                                   for i in range(num_contacts)]

                coll_data.append({
                    'agent_id': agent_id,
                    'patient_id': patient_id,
                    'relative_velocity': relative_velocity,
                    'impulse': impulse,
                    'num_contacts': num_contacts,
                    'contact_points': contact_points,
                    'contact_normals': contact_normals,
                    'state': state
                })
                if False:
                    print("agent: %d ---> patient %d" % (agent_id, patient_id))
                    print("relative velocity", relative_velocity)
                    print("impulse", impulse)
                    print("contact points", contact_points)
                    print("contact normals", contact_normals)
                    print("state", state)
            # collision between object and environment, not considered yet
            if  r_id == 'enco':
                collision = EnvironmentCollision(resp[i])
                # Not implemented yet
                pass
        return coll_data
