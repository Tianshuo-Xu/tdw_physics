#generate config combination
import os


scenario_name = "friction_platform"

template_config = f"/home/htung/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs/{scenario_name}_pp/commandline_args_template.txt"
prefix = f"{scenario_name}"


variables = dict()
exclude = dict()
variables["mass_dominoes"] = dict()
variables["mass_dominoes"]["num_middle_objects"] = [0, 1, 2, 3, 4]


variables["mass_waterpush"] = dict()
variables["mass_waterpush"]["target"] = ["bowl","cone","cube","cylinder","dumbbell","pentagon","pipe","pyramid"]
variables["mass_waterpush"]["tscale"] = ["0.5,0.5,0.5","0.45,0.5,0.45","0.4,0.5,0.4","0.35,0.5,0.35", "0.3,0.5,0.3"]
variables["mass_waterpush"]["zdloc"] = ["1","2","3"]
exclude["mass_waterpush"] = [{"target": ["cone"], "tscale":["0.5,0.5,0.5", "0.45,0.5,0.45","0.4,0.5,0.4"]}]


variables["mass_collision"] = dict()
variables["mass_collision"]["target"] = ["cone","cube","cylinder","dumbbell","pentagon","pipe","pyramid"]
variables["mass_collision"]["tscale"] = ["0.7,0.5,0.7", "0.6,0.5,0.6","0.5,0.5,0.5","0.4,0.5,0.4"]
variables["mass_collision"]["zdloc"] = ["1","2"]
exclude["mass_collision"] = []

variables["bouncy_platform"] = dict()
variables["bouncy_platform"]['use_blocker_with_hole'] = ["0", "1"]
variables["bouncy_platform"]["target"] = ["bowl","cone","cube","cylinder","dumbbell","pentagon","pipe","pyramid"]
variables["bouncy_platform"]["tscale"] = ["0.15,0.15,0.15", "0.2,0.2,0.2", "0.25,0.25,0.25", "0.3,0.3,0.3"]

variables["bouncy_wall"] = dict()
variables["bouncy_wall"]['zld'] = ["0", "2"]
variables["bouncy_wall"]["target"] = ["bowl","cone","cube","cylinder","dumbbell","pentagon","pipe","pyramid"]
variables["bouncy_wall"]["tscale"] = ["0.25,0.25,0.25", "0.35,0.35,0.35", "0.45,0.45,0.45"]



variables["friction_platform"] = dict()
variables["friction_platform"]["target"] = ["cone","cube","cylinder","dumbbell","pentagon","pipe","pyramid"]
variables["friction_platform"]["is_single_ramp"] = ["0","1"]
variables["friction_platform"]["zdloc"] = ["1","2"]
exclude["friction_platform"] = []



f = open(template_config, 'r')
template_content = f.read()


var_values =  [*variables[scenario_name].values()]
var_keys =  [*variables[scenario_name].keys()]
print("number of variables", len(var_keys))


def is_exclude(cur_list, ex_list):
    cur = dict()
    for key, val in cur_list:
        cur[key] = val
    for ex in ex_list:
        to_exclude = True
        for k, v in ex.items():
            if k not in cur:
                to_exclude = False
                break
            elif cur[k] not in v:
                to_exclude = False
                break
        if to_exclude:
            return True

def name_generator(vark, varv, cur_var_id, pfx, vars, ex):

    nvar = len(varv)
    for value in varv[cur_var_id]:
        cur_pfx = pfx + "-" + vark[cur_var_id] + "=" + f"{value}"
        cur_list = vars + [(vark[cur_var_id], value)]
        if cur_var_id == nvar - 1: #end
            #print(cur_pfx)
            #print(cur_list)
            is_ex = is_exclude(cur_list, ex)
            if is_ex:
                print("excluding", cur_pfx)
                print(cur_list)
                continue


            os.mkdir(cur_pfx)
            output_file = os.path.join(cur_pfx, "commandline_args.txt")
            print(output_file)

            fw = open(output_file, 'w')
            for variable, ele in cur_list:
                fw.write(f"--{variable}\n{ele}\n")
            fw.write(template_content)
            fw.close()

        else:
            name_generator(vark, varv, cur_var_id + 1, cur_pfx, cur_list, ex)

name_generator(var_keys, var_values, 0, os.path.join("/".join(template_config.split("/")[:-1]), prefix ), [], exclude[scenario_name])


# modify to dfs
# for variable, values in variables[scenario_name].items():
#     for ele in values:
#         output_dir = os.path.join("/".join(template_config.split("/")[:-1]), prefix + f"{variable}_{ele}")
#         print(output_dir)
#         #os.mkdir(output_dir)
#         #output_file = os.path.join(output_dir, "commandline_args.txt")
#         #print(output_file)

#         #fw = open(output_file, 'w')
#         #fw.write(f"--{variable}\n{ele}\n" + content)
#         #fw.close()

import ipdb; ipdb.set_trace()

print("hello")