import os
import socket
import glob
import h5py
import math
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt


import collections

#folder = "/media/htung/Extreme SSD/fish/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_3"
#folder="/home/hsiaoyut/2021/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_0"

def split_info(filename):
    infos = filename.split("-")[1:]
    info_dict = dict()
    for info in infos:
        arg_name, arg_value = info.split("=")
        info_dict[arg_name] = arg_value
    return info_dict

if "ccncluster" in socket.gethostname():
    data_root = "/mnt/fs4/hsiaoyut/physion++/analysis"
    #data_root = "/mnt/fs4/hsiaoyut/physion++/data_v1"
else:
    data_root = "/media/htung/Extreme SSD/fish/tdw_physics/dump_mini"

sname = "friction_platform_pp"
folder = os.path.join(data_root, sname)
#import ipdb; ipdb.set_trace()
filenames = os.listdir(folder)
restrict = "" #"bouncy_platform-use_blocker_with_hole=1" #"target_cone-tscale_0.35,0.5,0.35"
remove =  ["-is_single_ramp=1-zdloc=1"] #["pyramid", "cube", "cylinder", "pipe"] #["deform_clothhang-zdloc=2-target=pyramid", "deform_clothhang-zdloc=2-target=cube", "deform_clothhang-zdloc=1-target=cylinder"] #"-is_single_ramp=1-zdloc=1" #"simple_box1"
filenames = [filename for filename in filenames if restrict in filename]

target_varname = "star_friction" #"star_mass" #"star_mass", "star_deform"
merge_by = [] #["zdloc", "target"] #""#["is_single_ramp", "zdloc", 'pheight']#["zdloc", "target"]
#merge_by = "tscale"

set_dict = collections.defaultdict(list)

for filename in filenames:
    info_dict = split_info(filename)
    if merge_by == "all":
        set_dict["all"].append(filename)
    elif merge_by:
        set_dict[sname[:-3] + "-" + "-".join([mb + "=" +  info_dict[mb] for mb in merge_by])].append(filename)
    else:
        set_dict[filename].append(filename)



nbins = 8
means = []
set_names = []
for set_id, merge_var_name in enumerate(set_dict):
    target_params = []
    labels = []
    set_hist = []



    for filename in tqdm(set_dict[merge_var_name]):
        if remove and np.any([(rm in filename) for rm in remove]):
            continue

        pkl_id = 0
        pkl_filenames = [os.path.join(folder, filename, x) for x in os.listdir(os.path.join(folder, filename)) if x.endswith(".pkl")]
        for pkl_file in pkl_filenames: #glob.glob(os.path.join(folder, filename) + "/*.pkl"):
            print(pkl_file)
            with open(pkl_file, "rb") as f:
                f = pickle.load(f)

            #print(f['static']['cloth_material'])
            target_params.append(f["static"][target_varname])
            labels.append(float(f["static"]["does_target_contact_zone"]))
            pkl_id += 1


        print(labels)

    if len(labels) == 0:
        continue
    set_names.append(merge_var_name)

    # mass: 10^{-1} - 10^1 ==> -1 - 1 (after log)
    # bouncy: 0.01 - 1.0
    # frcition: 0.01 - 0.2
    # deform: 0 - 1

    if target_varname in ["star_mass"]:
        print(target_params)
        target_params = [math.log10(param) for param in target_params]
    elif target_varname in ["star_friction"]:
        target_params = [param * 2 * 5 - 1 for param in target_params]
    else:
        target_params = [param * 2 - 1 for param in target_params]

    #scatter plot
    #for oid, param in enumerate(target_params):
    #    if param > 0.5 and labels[oid] < 0.2:
    #        print("heavy object with negative outcome", oid)

    plt.scatter(target_params, labels)
    if target_varname in ["star_mass"]:
        plt.xlabel(f"log({target_varname})")
    else:
        plt.xlabel(f"{target_varname}")
    plt.ylabel("red hits yellow")
    ax = plt.gca()
    ax.set_ylim([-0.1, 1.6])



    n, be = np.histogram(target_params, bins=np.linspace(-1,1,nbins+1))
    sy, _ = np.histogram(target_params, bins=np.linspace(-1,1,nbins+1), weights=labels)
    mean = sy / (n + 0.0000001)

    # check nan and fill with interpolate

    means.append(mean)

    #std = np.sqrt(sy2/n - mean*mean)
    #plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='-', label=merge_var_name)


# if target_varname in ["star_mass"]:
#     target_params = [math.log10(param) for param in target_params]
# else:
#     target_params = [param for param in target_params]
# import matplotlib.pyplot as plt
# #scatter plot
# for oid, param in enumerate(target_params):
#     if param > 0.5 and labels[oid] < 0.2:
#         print("heavy object with negative outcome", oid)

# plt.scatter(target_params, labels)
# if target_varname in ["star_mass"]:
#     plt.xlabel(f"log({target_varname})")
# else:
#     plt.xlabel(f"{target_varname}")
# plt.ylabel("red hits yellow")
# ax = plt.gca()
# ax.set_ylim([-0.1, 1.5])


# nbins = 8
# n, _ = np.histogram(target_params, bins=nbins)

# sy, _ = np.histogram(target_params, bins=nbins, weights=labels)
# sy2, _ = np.histogram(target_params, bins=nbins, weights=labels)
# mean = sy / n
# std = np.sqrt(sy2/n - mean*mean)
# plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='-', label=merge_var_name)

    #hist plot
#plt.legend(loc="lower left")

means = np.stack(means, axis=1)

# A = means + 0.0000001
# b = 0.5 * np.ones_like(means[:,0])


#rom scipy.optimize import lsq_linear
#res = lsq_linear(A, b, bounds=(0, 1), lsmr_tol='auto', verbose=1)

# import cvxpy as cp
Q = means
b = 0.5 * np.ones_like(means[:,0]) # x dimension
# n = means.shape[1]

# x = cp.Variable(n)
# objective = cp.Minimize(cp.sum_squares(A @ x - b))
# constraints = [0 <= x, x <= 1, sum(x) == 1]
# prob = cp.Problem(objective, constraints)
# prob.solve()

# #Print result.
# print("\nThe optimal value is", prob.value)
# print("The optimal x is")
# print(x.value)
# print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)

# x is dimension 7
import cvxopt as cp
from cvxopt import matrix
mean_w = 10
mean_ = np.mean(means, axis=0)[np.newaxis, ...]
P_m = np.dot(mean_.T, mean_)
q_m =  -mean_[0] * 0.5# x dimension

P = Q.T @ Q
q = - Q.T.dot(b)
n = means.shape[1]

#G = np.concatenate([-np.eye(n), np.eye(n)], axis=0)
#h = np.concatenate([np.zeros(n), np.ones(n)])

min_w = (1.0 - 0.2)/(len(set_names)-1)
#min_w = 0.02
G = -np.eye(n)
h = (-min_w)* np.ones(n)

A = np.ones((1,n))
b = np.ones((1))


try:

    sol = cp.solvers.qp(matrix(P + mean_w * P_m), matrix(q + mean_w * q_m), matrix(G), matrix(h), matrix(A), matrix(b))
    print(set_names)
    print(sol['x'])
    new_mean = (Q@sol['x'])[:,0]
except:
    sol = dict()
    sol['x'] = (1.0/len(set_names)) * np.ones_like(np.mean(means, axis=0))
    new_mean = (Q@sol['x'])
import csv

with open(f"opt_weight/{sname}", 'w') as csvfile:
    csv_writer = csv.writer(csvfile)
    for set_id, set_name in enumerate(set_names):
        csv_writer.writerow([set_name, sol['x'][set_id], set_id])




#w = np.linalg.lstsq(means, 0.5 * np.ones_like(means[:,0]))
xaxis = (np.linspace(-1,1,nbins+1)[:nbins] + np.linspace(-1,1,nbins+1)[1:])/2.0
plt.plot(xaxis, np.mean(means, axis=1))
plt.plot(xaxis, new_mean)
plt.legend(["w/o opt", "w. opt"], loc="upper left")
plt.show()
rstr, mstr = "", ""
if restrict != "":
    rstr = f"_r{restrict}"
if merge_by != "":
    mname = ",".join(merge_by)
    mstr = f"_m{mname}"

plt.savefig(f"opt_s_curve_{sname}{rstr}{mstr}.png")

print(len(labels))
print(np.mean(new_mean), min_w)

#print(target_params)
