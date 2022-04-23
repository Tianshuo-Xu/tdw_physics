from tdw_physics.noisy.noisy_rigidbodies_dataset import RigidNoiseParams, NoisyRigidbodiesDataset
import json

noise_l = RigidNoiseParams(
    position={'x': .1, 'z': .1, 'y': None},
    rotation={'y': 25, 'x': None, 'z': None},
    bounciness=.1,
    mass=.1
)
noise_l.save('noise_low.json')

noise_h = RigidNoiseParams(
    position={'x': .2, 'z': .5, 'y': None},
    rotation={'y': 1, 'x': None, 'z': None},
    bounciness=2,
    mass=2,
    collision_mag=2
)
noise_h.save('noise_high.json')

noise_f = RigidNoiseParams(
    position={'x': .2, 'z': .5, 'y': .2},
    rotation={'y': 1, 'x': 1, 'z': 1},
    bounciness=2,
    mass=2,
    collision_mag=2
)
noise_f.save('noise_float.json')
