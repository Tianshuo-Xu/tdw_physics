import socket
import os
import csv
from pathlib import Path
import argparse
import subprocess
import numpy as np
parser = argparse.ArgumentParser(description='Data generation for physion++.')
parser.add_argument("--scenario",
                    type=str,
                    default="mass_dominoes",
                    help="scaneraio name, e.g., mass_dominoes, friction_platform")
parser.add_argument("--total_nums",
                    type=int,
                    default=200,
                    help="total number of element to generate from the scene")
parser.add_argument("--gpu",
                    type=int,
                    default=0,
                    help="total number of element to generate from the scene")
parser.add_argument("--st",
                    type=int,
                    default=0,
                    help="total number of element to generate from the scene")
parser.add_argument("--ed",
                    type=int,
                    default=100,
                    help="total number of element to generate from the scene")


args = parser.parse_args()

OPT_CONFIG="./opt_weight"
if socket.gethostname() == "aw-m17-R2":
    PHYSICS_HOME = str(Path.home()) + "/Documents/2021"
    DUMP_DIR = "/media/htung/Extreme SSD/fish/tdw_physics/dump_mini4"
elif "ccncluster" in socket.gethostname():
    PHYSICS_HOME = str(Path.home()) + "/2021"
    DUMP_DIR = "/mnt/fs4/hsiaoyut/physion++/data_v1"
else:
    raise ValueError

ARGS_PATH=os.path.join(PHYSICS_HOME, "physics-benchmarking-neurips2021-htung/stimuli/generation/configs")
image_size = 256
gpu = args.gpu
scenario = args.scenario

controller = dict()
controller["mass_dominoes"] = "tdw_physics/target_controllers/dominoes_var.py"
controller["mass_waterpush"] = "tdw_physics/target_controllers/waterpush_var.py"
controller["mass_collision"] = "tdw_physics/target_controllers/collision_var.py"
controller["bouncy_platform"] = "tdw_physics/target_controllers/bouncy.py"
controller["bouncy_wall"] = "tdw_physics/target_controllers/bouncywall_var.py"
controller["friction_platform"] = "tdw_physics/target_controllers/fricramp.py"
controller["friction_collision"] = "tdw_physics/target_controllers/collisionfric_var.py"
controller["friction_cloth"] = "tdw_physics/target_controllers/fricrampcloth.py"
controller["deform_clothhang"] = "tdw_physics/target_controllers/clothhang_var.py"
controller["deform_clothhit"] = "tdw_physics/target_controllers/clothhit_var.py"
controller["deform_clothdrop"] = "tdw_physics/target_controllers/dropcloth_var.py"



# read arg_name, weight and seed from config file
with open(os.path.join(OPT_CONFIG, scenario + "_pp")) as csvfile:
    rows = csv.reader(csvfile)
    out = [row for row in rows]

total_nums = args.total_nums
res = args.total_nums
total_weight = 0
gen_nums = dict()
for arg, weight, seed in out:
    weight = float(weight)
    seed = int(seed)
    num = round(weight * total_nums)
    gen_nums[arg] = num
    res -= num
    total_weight += weight

print(gen_nums)
print("before:", np.sum([v for k, v in gen_nums.items()]))
out2 = sorted(out, key=lambda x: x[1], reverse=True)
nargs = len(out)
if res >= 0:
    nargs = len(out)
    for i in range(res):
        arg, w, s = out2[i % nargs]
        gen_nums[arg] += 1

else:
    for i in range(-res):
        arg, w, s = out2[i % nargs]
        gen_nums[arg] -= 1
        print("delete", arg)

print("after:", np.sum([v for k, v in gen_nums.items()]))
assert(abs(total_weight - 1) < 0.01)

#out = [k for k in out if "zdloc=1" in k[0]]
#import ipdb; ipdb.set_trace()
total_nums = args.total_nums
cur_num = 0
weight_sum = 0
#print("processing:", [k[2] for k in out[args.st: args.ed]])
#import ipdb; ipdb.set_trace()
for arg, weight, seed in out[args.st:args.ed]:
    weight = float(weight)
    seed = int(seed)
    #num = round(weight * total_nums)
    num = gen_nums[arg]

    args_f = os.path.join(ARGS_PATH, scenario + "_pp",  arg) + "/commandline_args.txt"
    dump_dir = os.path.join(DUMP_DIR, scenario + "_pp", arg).replace(" ", "\ ")

    # if scenario == "deform_clothhang":
    #     cmd = f"CUDA_VISIBLE_DEVICES={gpu} python {controller[scenario]} @{args_f} --dir {dump_dir}  --debug_data_mode --height 64 --width 64 --gpu {gpu} --num 30 --seed {seed}"

    #else:
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} python {controller[scenario]} @{args_f} --dir {dump_dir}  --training_data_mode --height {image_size} --width {image_size} --gpu {gpu} --num {num} --seed {seed}"
    os.system(cmd)
    print(cmd)
    cur_num += num
    weight_sum += weight
    print(weight, weight_sum, cur_num)
import ipdb; ipdb.set_trace()
print(cur_num, "/", total_nums)





