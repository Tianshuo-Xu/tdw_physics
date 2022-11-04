import socket
import os
import os.path
import csv
from pathlib import Path
import argparse
import subprocess
import numpy as np
import math
import pickle
import shutil
from datetime import date
parser = argparse.ArgumentParser(description='Data generation for physion++.')
parser.add_argument("--scenario",
                    type=str,
                    default="mass_dominoes",
                    help="scaneraio name, e.g., mass_dominoes, friction_platform")
parser.add_argument("--total_nums",
                    type=int,
                    default=64,
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
    DUMP_DIR = "/media/htung/Extreme SSD/fish/tdw_physics/dump_1102"
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

exclude = dict()
exclude["mass_dominoes"] = [{"remove_middle":[1]}]
exclude["mass_waterpush"] = [{"target": ["cone"], "tscale":["0.5,0.5,0.5", "0.45,0.5,0.45","0.4,0.5,0.4"]}]
exclude["mass_collision"] = []

additional_cmd = dict()
# 45
additional_cmd["mass_collision"] = "--camera_min_angle 15 --camera_max_angle 75 --camera_min_height 0.75 --camera_max_height 2.0 --zjitter 0.2"
additional_cmd["mass_waterpush"] = "--camera_min_angle 15 --camera_max_angle 75 --camera_min_height 0.3 --camera_max_height 0.75 --zjitter 0.2 --save_labels"
def is_exclude(cur_list, ex_list):
    cur = dict()
    for key, val in cur_list:
        cur[key] = val
    for ex in ex_list:
        to_exclude = True
        for k, v in ex.items():
            #print(k, v, cur)
            if k not in cur:
                #print("here1")
                to_exclude = False
                break
            elif cur[k] not in [str(vv) for vv in v]:
                #print("here2")
                to_exclude = False
                break
        if to_exclude:
            return True

def mkdir(path, is_assert=False):
    if is_assert:
        assert(not os.path.exists(path)), f"{path} exists, delete it first if you want to overwrite"
    if not os.path.exists(path):
        os.makedirs(path)


# read arg_name, weight and seed from config file
with open(os.path.join(OPT_CONFIG, scenario + "_pp")) as csvfile:
    rows = csv.reader(csvfile)
    csv_data = [r for r in rows]
    print("total num of rows", len(csv_data))
    ex_list = exclude[scenario]
    #out = [row for row in rows]
    out = []
    for row in csv_data:
        cur_list = [l.split("=") for l in row[0].split("-")[1:]]
        if not is_exclude(cur_list, ex_list):
            out.append(row)

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


# file for storing the generated output
f = open(os.path.join(DUMP_DIR, scenario + "_pp_sampling.csv"), 'a')
record_writer = csv.writer(f)

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
#assert(abs(total_weight - 1) < 0.01)

#out = [k for k in out if "zdloc=1" in k[0]]
#import ipdb; ipdb.set_trace()
total_nums = args.total_nums
cur_num = 0
weight_sum = 0
#print("processing:", [k[2] for k in out[args.st: args.ed]])
#import ipdb; ipdb.set_trace()


def norm_phvar_to_value(input_, scenario):
    # norm is from [0,1]
    if scenario.startswith("mass_"):
        return 10**(input_ * 2 - 1)
    elif scenario.startswith("friction_"):
        return 0.2 * input_
    else:
        return input_

def phvar_to_norm_value(input_, scenario):
    # norm is from [0,1]
    if scenario.startswith("mass_"):
        return 0.5 * (math.log10(input_) + 1) #10**(input_ * 2 - 1)
    elif scenario.startswith("friction_"):
        return input_ * 5
    else:
        return input_

def get_label_from_pkl(filename, return_f=False):
    with open(filename, "rb") as f:
        f = pickle.load(f)
    if return_f:
        return f["static"]["does_target_contact_zone"], f
    else:
        return f["static"]["does_target_contact_zone"]

file_types = [".pkl", "_id.json", "_img.mp4", "_id.mp4", "_map.png"]
MAX_SAMPLE = 5
SAMPLES_TO_CHECK_S_CURVE = 2

for arg, weight, seed in out[args.st:args.ed]:

    weight = float(weight)
    seed = int(seed)
    #num = round(weight * total_nums)
    num = gen_nums[arg]

    # DATA PATH
    args_f = os.path.join(ARGS_PATH, scenario + "_pp",  arg) + "/commandline_args.txt"
    dump_dir = os.path.join(DUMP_DIR, scenario + "_pp", arg)
    dump_copy0_dir = os.path.join(DUMP_DIR, scenario + "_pp-copy0", arg)
    dump_copy1_dir = os.path.join(DUMP_DIR, scenario + "_pp-copy1", arg)
    mkdir(dump_copy0_dir)
    mkdir(dump_copy1_dir)

    dump_dir_cmd = dump_dir.replace(" ", "\ ")
    add_cmd = additional_cmd[scenario] if scenario in additional_cmd else ""
    base_cmd = f"CUDA_VISIBLE_DEVICES={gpu} python {controller[scenario]} @{args_f} {add_cmd} --dir {dump_dir_cmd}  --testing_data_mode --height {image_size} --width {image_size} --gpu {gpu} --seed {seed}"

    var_rng_seed = (seed * 1113) % 1993

    # compute exists up to
    exists_up_to = -1

    all_copy0_files = os.listdir(dump_copy0_dir)
    all_copy1_files = os.listdir(dump_copy1_dir)
    for i in range(num):
        missing_trial = False
        for file_type in file_types:

            if not os.path.exists(os.path.join(dump_copy0_dir, f"{i:04}{file_type}")) or not os.path.exists(os.path.join(dump_copy1_dir, f"{i:04}{file_type}")):
                missing_trial = True
                break
                #print("id", i, "here2", [os.path.exists(os.path.join(dump_dir, f"{i:04}_{j:04}.pkl")) for j in range(1,1+MAX_SAMPLE)])
                #print(os.path.join(dump_copy0_dir, f"{i:04}{file_type}"))
                #if not all([os.path.exists(os.path.join(dump_dir, f"{i:04}_{j:04}.pkl")) for j in range(1,1+MAX_SAMPLE)]):
                #    missing_trial = True
                #    break
        print(os.path.join(dump_copy0_dir, f"{i:04}{file_type}"))
        print("id", i, f"/{num}, missing:", missing_trial)
        if not missing_trial:
            label, f = get_label_from_pkl(os.path.join(dump_copy1_dir, f"{i:04}.pkl"), return_f=True)
            print(f"trial_{i}",  phvar_to_norm_value(f["static"]["star_" + scenario.split("_")[0]], scenario), label)
            exists_up_to = i
        else:
            exists_up_to = i-1
            break
    #import ipdb; ipdb.set_trace()

    if exists_up_to + 1 == num:
        continue
    label_when_using_max_phyvar = []
    if exists_up_to > 0:
        for trial_num in range(min(exists_up_to + 1, SAMPLES_TO_CHECK_S_CURVE)):
            label = get_label_from_pkl(os.path.join(dump_copy0_dir, f"{trial_num:04}.pkl"))
            label_when_using_max_phyvar.append(label)

    offset = 0
    for trial_num in range(exists_up_to + 1, num):
        var_rng = np.random.RandomState(var_rng_seed + trial_num + offset)
        sample_minb = 0
        sample_maxb = 1
        current_label = None
        picked_trials = []
        target_params = []
        labels = []

        # evaluate the right most value
        if len(label_when_using_max_phyvar) < SAMPLES_TO_CHECK_S_CURVE:
            sub_id = 0
            if not all([os.path.exists(os.path.join(dump_dir, f"{trial_num:04}_{sub_id:04}{file_type}")) for file_type in file_types]):
                phyvar = norm_phvar_to_value(1, scenario)
                cmd = base_cmd + f" --trial_id {trial_num} --sub_id {sub_id} --phy_var {phyvar}"
                print("run cmd", cmd)
                os.system(cmd)
            else:
                print("skipping generation of ", os.path.join(dump_dir, f"{trial_num:04}_{sub_id:04}"))

            label = get_label_from_pkl(os.path.join(dump_dir, f"{trial_num:04}_{sub_id:04}.pkl"))
            is_s_curve = label # s curve
            label_when_using_max_phyvar.append(label)
        else:
            is_s_curve = np.mean(label_when_using_max_phyvar) > 0.5


        exists_label = None
        for sample_id in range(1, MAX_SAMPLE+1):
            if var_rng.uniform(0.0, 1.0) > 0.5:
                print("sample st1:", sample_minb, sample_maxb)
                nphyvar = var_rng.uniform(sample_minb, sample_maxb)
            else: # randomly sample with some probabilitic
                print("sample st2:", sample_minb, sample_maxb)
                nphyvar = var_rng.uniform(0.0, 1.0)

            if not all([os.path.exists(os.path.join(dump_dir, f"{trial_num:04}_{sample_id:04}{file_type}")) for file_type in file_types]):
                phyvar = norm_phvar_to_value(nphyvar, scenario)
                cmd = base_cmd + f" --trial_id {trial_num} --sub_id {sample_id} --phy_var {phyvar}"
                label = phyvar > 0.5
                os.system(cmd)
                print(cmd)
            else:
                print("skipping generation of ", os.path.join(dump_dir, f"{trial_num:04}_{sample_id:04}"))
            # read trial value
            current_label, f = get_label_from_pkl(os.path.join(dump_dir, f"{trial_num:04}_{sample_id:04}.pkl"), return_f=True)

            #pkl_filename = os.path.join(dump_dir, f"{trial_num:04}.pkl")
            #with open(pkl_filename, "rb") as f:
            #    f = pickle.load(f)
            print("exists_label", exists_label)
            print("picked_trials", picked_trials)
            print("current_label", current_label)
            nnphyvar = phvar_to_norm_value(f["static"]["star_" + scenario.split("_")[0]], scenario)
            target_params.append(nnphyvar)
            labels.append(current_label)
            if sample_id == 1:
                picked_trials.append(sample_id)
                #current_label = float(f["static"]["does_target_contact_zone"])
                if float(is_s_curve) == current_label:
                    sample_maxb = nnphyvar
                else:
                    sample_minb = nnphyvar
                exists_label = current_label

            elif exists_label == (1-current_label):
                picked_trials.append(sample_id)
                break


        print(target_params)
        print(labels)
        date_str = date.today().strftime("%y%m%d")
        record_writer.writerow([dump_dir, trial_num] + [offset, date_str] + labels + [None] * (MAX_SAMPLE - len(labels)) + target_params + [None] * (MAX_SAMPLE - len(labels)))
        if len(picked_trials) < 2:
            # still save the bad ones
            for file_type in file_types:
                shutil.copyfile(os.path.join(dump_dir, f"{trial_num:04}_{picked_trials[0]:04}{file_type}"), os.path.join(dump_copy0_dir, f"{trial_num:04}{file_type}"))
                shutil.copyfile(os.path.join(dump_dir, f"{trial_num:04}_{sample_id:04}{file_type}"), os.path.join(dump_copy1_dir, f"{trial_num:04}{file_type}"))


            offset += 1
            continue
        # copy file
        #for sub_id in enumerate(picked_trials):
        for file_type in file_types:
            shutil.copyfile(os.path.join(dump_dir, f"{trial_num:04}_{picked_trials[0]:04}{file_type}"), os.path.join(dump_copy0_dir, f"{trial_num:04}{file_type}"))
            shutil.copyfile(os.path.join(dump_dir, f"{trial_num:04}_{picked_trials[1]:04}{file_type}"), os.path.join(dump_copy1_dir, f"{trial_num:04}{file_type}"))



    cur_num += num
    weight_sum += weight
    print(weight, weight_sum, cur_num)
    # delete after everything is done
    file_to_remove = [os.path.join(dump_dir, x) for x in os.listdir(dump_dir)]
    print("are you sure to delete the following files under {dump_dir}?", file_to_remove)
    [os.remove(x) for x in file_to_remove]

import ipdb; ipdb.set_trace()
print(cur_num, "/", total_nums)


### todo
## add exists-up-to for trial_num and sub_id
## overlay video and run through aws3 upload



