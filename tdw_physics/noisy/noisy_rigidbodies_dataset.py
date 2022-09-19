from abc import ABC
import numpy as np
import random
from tdw.librarian import ModelRecord
from tdw_physics.rigidbodies_dataset import RigidbodiesDataset
from tdw.output_data import OutputData, Rigidbodies, Collision, EnvironmentCollision
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import copy
from scipy.stats import vonmises, norm, lognorm
from tdw_physics.noisy.noisy_utils import rotmag2vec, vec2rotmag,\
        rad2deg, deg2rad
import os
import json
import warnings

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
    """

    position: Dict[str, float] = None
    rotation: Dict[str, float] = None
    velocity_dir: Dict[str, float] = None
    velocity_mag: float = None
    mass: float = None
    static_friction: float = None
    dynamic_friction: float = None
    bounciness: float = None
    collision_dir: Dict[str, float] = None
    collision_mag: float = None

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
            'collision_mag': self.collision_mag
        }
        with open(flpth, 'w') as ofl:
            json.dump(selfobj, ofl)

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
        if collision_dir is not None:
            collision_dir = dict([[k, collision_dir] for k in XYZ])
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
                 **kwargs):
        RigidbodiesDataset.__init__(self, **kwargs)
        self._noise_params = noise

        # how to generate collision noise
        self._ongoing_collisions = []
        self._lasttime_collisions = []
        self.set_collision_noise_generator(noise)
        if self.collision_noise_generator is not None:
            print("example noise", self.collision_noise_generator())

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
        Overwrites method from rigidbodies_dataset to add noise to objects when added to the scene
        """
        n = self._noise_params
        rotrad = dict([[k, deg2rad(rotation[k])]
                       for k in rotation.keys()])
        for k in XYZ:
            if n.position is not None and k in n.position.keys()\
                    and n.position[k] is not None:
                position[k] = norm.rvs(position[k],
                                       n.position[k])
            if n.rotation is not None and k in n.rotation.keys()\
                    and n.rotation[k] is not None:
                rotrad[k] = vonmises.rvs(n.rotation[k], rotrad[k])
        rotation = dict([[k, rad2deg(rotrad[k])]
                         for k in rotrad.keys()])
        if n.mass is not None:
            mass = lognorm.rvs(n.mass, mass)
        # Clamp frictions to be > 0
        if n.dynamic_friction is not None:
            dynamic_friction = max(0,
                                   norm.rvs(dynamic_friction,
                                            n.dynamic_friction))
        if n.static_friction is not None:
            static_friction = max(0,
                                  norm.rvs(static_friction,
                                           n.static_friction))
        # Clamp bounciness between 0 and 1
        if n.bounciness is not None:
            bounciness = max(0, min(1,
                                    norm.rvs(bounciness,
                                             n.bounciness)))
        return RigidbodiesDataset.add_physics_object(
            self,
            record, position, rotation, mass,
            dynamic_friction, static_friction,
            bounciness, o_id, add_data
        )

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:
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
                new_collisions = []
                for cd in coll_data:
                    # Ensure consistency in naming / comparisons
                    aid = min(cd['agent_id'], cd['patient_id'])
                    pid = max(cd['agent_id'], cd['patient_id'])
                    nm_ap = str(aid) + '_' + str(pid)

                    """ This seems inefficient... """
                    for r in resp[:-1]:
                        r_id = OutputData.get_data_type_id(r)
                        if r_id == "rigi":
                            ri = Rigidbodies(r)
                            for i in range(ri.get_num()):
                                if ri.get_id(i) == aid:
                                    va = ri.get_velocity(i)
                                elif ri.get_id(i) == pid:
                                    vp = ri.get_velocity(i)

                    if cd['state'] == 'enter':
                        print('start rvel: ' + nm_ap + ' : '
                              + str(va) + '; ' + str(vp))
                        self._ongoing_collisions.append(nm_ap)
                        new_collisions.append(nm_ap)
                    else:
                        if nm_ap in self._lasttime_collisions:
                            print('next rvel: ' + nm_ap + ' : '
                                  + str(va) + '; ' + str(vp))

                    if cd['state'] == 'exit':
                        self._ongoing_collisions = [
                            c for c in self._ongoing_collisions if
                            c != nm_ap
                        ]
                self._lasttime_collisions = new_collisions
                #coll_noise_cmds = self.apply_collision_noise(resp, coll_data)
                #cmds.extend(coll_noise_cmds)

        return cmds

    def apply_collision_noise(self, resp, data=None):
        if data is None:
            return []
        o_id = data['patient_id']
        force = self.collision_noise_generator()
        #print("collision noise", force)
        cmds = [
            {
                "$type": "apply_force_to_object",
                "force": force,
                "id": o_id
            }
        ]
        return cmds

    """ INCOMPLETE - Adds force but not relative """

    """
    Helper function that takes in a collision momenum transfer vector, then calculates the momentum to apply
    in the next timestep to actualize the collision noise
    """
    def _calculate_collision_differential(self, momentum: Dict[str, float]) -> Dict[str, float]:
        warnings.warn('NOT IMPLEMENTED YET - ADDS CONSTANT 1,1,1 MOMENTUM')
        return {'x':1, 'y': 1, 'z': 1}



    def set_collision_noise_generator(self,
                                      noise_obj: RigidNoiseParams):
        # Only make noise if there is noise to be added
        if noise_obj.collision_dir is not None:
            ncd = copy.copy(noise_obj.collision_dir)
        else:
            ncd = None
        ncm = noise_obj.collision_mag
        if (ncd is None
                or any([k in ncd.keys() for k in XYZ])) and\
                ncm is None:
            self.collision_noise_generator = None
        else:
            """ NOTE MAKE THIS ALL RELATIVE | ADD INPUT """
            if ncd is None:

                def cng():
                    return rotmag2vec(dict([[k, 0] for k in XYZ]),
                                      lognorm.rvs(ncm, 1))
            elif ncm is None:
                def cng():
                    return rotmag2vec(dict([[k, vonmises.rvs(ncd[k], 0)]
                                            for k in XYZ]),
                                      1)
            else:
                def cng():
                    return rotmag2vec(dict([[k, vonmises.rvs(ncd[k], 0)]
                                            for k in XYZ]),
                                      lognorm.rvs(ncm, 1))
            self.collision_noise_generator = cng

    def settle(self):
        """
        After adding a set of objects in a noisy way,
        'settles' the world to avoid interpenetration
        """
        raise NotImplementedError()

    """ Ensures collision data is sent pre (change for post) """

    def _get_send_data_commands(self) -> List[dict]:
        commands = super()._get_send_data_commands()
        # Can't send this more than once...
        commands = [c for c in commands if c['$type'] != 'send_collisions']
        commands.extend([{"$type": "send_collisions",
                          "enter": True,
                          "exit": False,
                          "stay": False,
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
                num_contacts = co.get_num_contacts()
                contact_points = [co.get_contact_point(i)
                                  for i in range(num_contacts)]
                contact_normals = [co.get_contact_normal(i)
                                   for i in range(num_contacts)]

                coll_data.append({
                    'agent_id': agent_id,
                    'patient_id': patient_id,
                    'relative_velocity': relative_velocity,
                    'num_contacts': num_contacts,
                    'contact_points': contact_points,
                    'contact_normals': contact_normals,
                    'state': state
                })
                #if self.PRINT:
                if False:
                    print("agent: %d ---> patient %d" % (agent_id, patient_id))
                    print("relative velocity", relative_velocity)
                    print("contact points", contact_points)
                    print("contact normals", contact_normals)
                    print("state", state)

        return coll_data
