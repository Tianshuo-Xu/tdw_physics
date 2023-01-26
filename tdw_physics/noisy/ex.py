import os
import h5py
import io
from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import display, clear_output
import pandas as pd
from numpy import linalg as LA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def plot_3d_scatter(data,ax=None,colour='red',sz=30,el=20,az=50,sph=True,sph_colour="gray",sph_alpha=0.001,
                    eq_line=True,pol_line=True,grd=False):
    ax.view_init(el, az)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_zlim(-1.5,1.5)

    # Add a shaded unit sphere
    if sph:
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color=sph_colour,alpha=sph_alpha)

    # Add an equitorial line
    if eq_line:
        # t = theta, p = phi
        eqt = np.linspace(0,2*np.pi,50,endpoint=False)
        eqp = np.linspace(0,2*np.pi,50,endpoint=False)
        eqx = 2*np.sin(eqt)*np.cos(eqp)
        eqy = 2*np.sin(eqt)*np.sin(eqp) - 1
        eqz = np.zeros(50)

        # Equator line
        ax.plot(eqx,eqy,eqz,color="k",lw=1)

    # Add a polar line
    if pol_line:
        # t = theta, p = phi
        eqt = np.linspace(0,2*np.pi,50,endpoint=False)
        eqp = np.linspace(0,2*np.pi,50,endpoint=False)
        eqx = 2*np.sin(eqt)*np.cos(eqp)
        eqy = 2*np.sin(eqt)*np.sin(eqp) - 1
        eqz = np.zeros(50)

        # Polar line
        ax.plot(eqx,eqz,eqy,color="k",lw=1)

    # Draw a centre point
    ax.scatter([0], [0], [0], color="k", s=sz)    

    # Turn off grid
    ax.grid(grd)

    # Ticks
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    ax.set_zticks([-1,0,1])
    
    return ax.scatter(data[:,0],data[:,1],data[:,2],s=1,c=colour)

def plot_arrow(point,ax,colour="red"):
    # Fancy arrow 
    a = Arrow3D([0, point[0]], [0, point[1]], [0, point[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color=colour)
    return ax.add_artist(a)

# Drawing a fancy vector see Ref. [7] 
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

proj_dir = os.path.abspath('.')
controllers_dir =  os.path.join(proj_dir,'controllers')
# specify objects
obj_idxs = [[3,4], [4,5], [5,6], [2,6], [1,2]]
obj_color = {'3-4': 'r', '4-5': 'b', '5-6': 'green', '2-6': 'orange', '1-2': 'purple', '4-3': 'r', '5-4': 'b', '6-5': 'green', '6-2': 'orange', '2-1': 'purple'}
num_trial = 20
fs = 16
# read ref file:
ref_dir =  os.path.join(controllers_dir,'tmp_ref')
data_dirs = [os.path.join(controllers_dir,'tmp_h')]

for obj_idx in obj_idxs:
    print(obj_idx)
    for data_dir in data_dirs:
        # init figure
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        img = []

        file_path = os.path.join(ref_dir, f"{0:04}"+'.hdf5')
        h5_file = h5py.File(file_path, 'r')

        collisions_ref = []
        with h5_file as f:
            frames = list(f['frames'])
            for j, frame in enumerate(frames):
                for i, object_id_tuple in enumerate(f['frames'][frame]['collisions']['object_ids'][:]):
                    if set(obj_idx) == set(object_id_tuple):
                        impulse = f['frames'][frame]['collisions']['impulses'][:][i]
                        if np.array_equal(impulse,(0,0,0)):
                            collisions_ref.append((0,0,0))
                        else:
                            collisions_ref.append(impulse/np.linalg.norm(impulse))

        # read noisy file:
        collisions = {}
        for trial_index in range(0, num_trial):
            file_path = os.path.join(data_dir, f"{trial_index:04}"+'.hdf5')
            h5_file = h5py.File(file_path, 'r')
            with h5_file as f:
                frames = list(f['frames'])
                for j, frame in enumerate(frames):
                    if j < len(collisions_ref):
                        for i, object_id_tuple in enumerate(f['frames'][frame]['collisions']['object_ids'][:]):
                            if set(obj_idx) == set(object_id_tuple):
                                object_id_str = '-'.join([str(x) for x in object_id_tuple])
                                impulse = f['frames'][frame]['collisions']['impulses'][:][i]
                                if str(trial_index) not in collisions.keys():
                                    collisions[str(trial_index)] = []
                                else:
                                    if np.array_equal(impulse, (0,0,0)):
                                        collisions[str(trial_index)].append((0,0,0))
                                    else:
                                        if np.dot(impulse, collisions_ref[j]) < 0:
                                            impulse = np.negative(impulse)
                                        collisions[str(trial_index)].append(impulse/np.linalg.norm(impulse))

        all_lengths = [len(x) for x in collisions.values()]
        all_lengths.append(len(collisions_ref))
        length = min(all_lengths)

        for frame_idx in range(length):
            impulse = collisions_ref[frame_idx]
            noisy_impulse = np.array([x[frame_idx] for x in collisions.values()])
            img.append([plot_arrow(impulse,ax1,colour=obj_color[object_id_str]), plot_3d_scatter(noisy_impulse, ax1, colour=obj_color[object_id_str], sph_alpha=0.0019)])

        # Labels
        ax1.set_xlabel('x',fontsize=fs)
        ax1.set_ylabel('y',fontsize=fs)
        ax1.set_zlabel('z',fontsize=fs)

        print('saving direction!')
        ani = animation.ArtistAnimation(fig, img, interval=50, blit=True, repeat=True, repeat_delay=1000)
        ani.save(data_dir+'/direction_'+object_id_str+'.gif', writer='pillow', fps=10, dpi=300, progress_callback = lambda i, n: print(f'Saving frame {i} of {n}'))
        print('direction finished!')
