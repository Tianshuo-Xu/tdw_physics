from hyperopt import hp
from hyperopt import fmin, tpe
import hyperopt
from extract_labels import *
import os
import h5py
from noisy_rigidbodies_dataset import RigidNoiseParams


# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(hyperopt.space_eval(space, best))
# -> ('case 2', 0.01420615366247227}


proj_dir = os.path.abspath('.')
controllers_dir =  os.path.join(proj_dir,'controllers')

def extract_init_params(d):
    mass = get_static_val(d, 'mass')
    init_pos = get_static_val(d, 'initial_position')
    init_rot = get_static_val(d, 'initial_rotation')
    model_name = get_static_val(d, 'model_names')
    bounciness = get_static_val(d, 'bounciness')
    dynamic_friction = get_static_val(d, 'dynamic_friction')
    return {}

def set_init_params():
    return

def run_simulation(num, noise_param, init_param):
    noise_param.save(os.path.join(controllers_dir,f'noise_{noise_param.to_string()}.json'))
    data_dir = os.path.join(controllers_dir, f'tmp_{noise_param.to_string()}')
    os.system(f"python3 noisy_dominoes.py --dir {data_dir} --num {num} --height 1 --width 1 --framerate 60 --noise noise_{noise_param.to_string()}.json -- init_param {init_param}.json")
    paths = [os.path.join(data_dir, f"{idx:04}"+'.hdf5') for idx in range(num)]
    fs = [h5py.File(path, mode='r') for path in paths]
    acc_avg = avg_label(list(map(does_target_contact_zone, fs)))
    for f in fs:
        f.close()
    return

noise_param = RigidNoiseParams(
    position={'x': .1, 'z': .1, 'y': None},
    rotation={'y': 25, 'x': None, 'z': None},
    bounciness=.1,
    mass=.1
)
