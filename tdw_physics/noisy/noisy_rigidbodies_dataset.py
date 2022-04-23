from abc import ABC
import numpy as np
import random
from tdw.librarian import ModelRecord
from tdw_physics.rigidbodies_dataset import RigidbodiesDataset
from dataclass import dataclass
from typing import List, Tuple, Dict, Optional
import copy
from scipy.stats import vonmises, norm, lognorm
from noisy_utils import rotmag2vec


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
XYZ = ['x', 'y', 'z']


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
        #print("noise range", collision_noise_range)
        self.set_collision_noise_generator(noise)
        if self.collision_noise_generator is not None:
            print("example noise", self.collision_noise_generator())

    """ TODO: MAKE GAUSSIAN """

    def set_collision_noise_generator(self,
                                      noise_obj: RigidNoiseParams):
        # Only make noise if there is noise to be added
        ncd = copy.copy(noise_obj['collision_dir'])
        ncm = noise_obj['collision_mag']
        if (ncd is None
                or any([k in ncd.keys() for k in XYZ])) and\
                ncm is None:
            self.collision_noise_generator = None
        else:
            def f():
                pass
        """
        if noise_range is None:
            self.collision_noise_generator = None
        elif hasattr(noise_range, 'keys'):
            assert all([k in noise_range.keys() for k in ['x', 'y', 'z']])
            self.collision_noise_generator = lambda: {
                'x': random.uniform(noise_range['x'][0], noise_range['x'][1]),
                'y': random.uniform(noise_range['y'][0], noise_range['y'][1]),
                'z': random.uniform(noise_range['z'][0], noise_range['z'][1])
            }
        elif hasattr(noise_range, '__len__'):
            assert len(noise_range) == 2, len(noise_range)
            self.collision_noise_generator = lambda: {
                'x': random.uniform(noise_range[0], noise_range[1]),
                'y': random.uniform(noise_range[0], noise_range[1]),
                'z': random.uniform(noise_range[0], noise_range[1])
            }
        else:
            raise ValueError("%s cannot be interpreted as a noise range" \
                             % noise_range)
        """

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
        n = self.noise
        for k in XYZ:
            if n.position is not None and k in n.position.keys():
                position[k] = norm.rvs(position[k],
                                       n.position[k])
            if n.rotation is not None and k in n.rotation.keys():
                rotation[k] = vonmises.rvs(n.rotation[k],
                                           rotation[k])
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
            record, position, rotation, mass,
            dynamic_friction, static_friction,
            bounciness, o_id, add_data
        )

    def settle(self):
        """
        After adding a set of objects in a noisy way,
        'settles' the world to avoid interpenetration
        """
        raise NotImplementedError()
