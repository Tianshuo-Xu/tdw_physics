import h5py
import ipdb
st=ipdb.set_trace

file="log2/0000.hdf5"

f = h5py.File(file, "r")
# [name for name in f["frames"]['0000']]  
# ['camera_matrices', 'collisions', 'env_collisions', 'images', 'labels', 'objects']

## camera
# camera_matrices: ['camera_matrix', 'projection_matrix']
# f["frames"]['0000']["camera_matrices"]["camera_matrix"][:]: 16
# f["frames"]['0000']["camera_matrices"]["projection_matrix"][:]: 16

## objects
# [name for name in f["frames"]['0000']["objects"]]
# ['angular_velocities', 'forwards', 'positions', 'rotations', 'velocities']
# f["frames"]['0000']["objects"]["positions"][:]: #objects x 3
# f["frames"]['0000']["objects"]["rotations"][:]: #objects x 4
# f["frames"]['0000']["objects"]["velocities"][:]: #objects x 3
# f["frames"]['0000']["objects"]["forwards"][:]: #objects x 3


## name
#['bounciness', 'color', 'distractors', 'dynamic_friction', 'git_commit', 'initial_position', 'initial_rotation', 'mass', 'middle_objects', 'middle_type', 'model_names', 'object_ids',
#'object_segmentation_colors', 'occluders', 'probe_id', 'probe_mass', 'probe_type', 'push_force', 'push_position', 'push_time', 'randomize', 'remove_middle', 'room', 'scale', 'scale_x', 'scale_y', 'scale_z', 'seed', 'static_friction', 'stimulus_name', 'target_id', 'target_rotation', 'target_type', 'trial_num', 'trial_seed', 'zone_id']


st()
print("hello")

