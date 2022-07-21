#generate config combination
import os


scenario_name = "mass_waterpush"

template_config = f"/home/htung/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs/{scenario_name}_pp/commandline_args_template.txt"
prefix = f"{scenario_name}"


variables = dict()
variables["mass_dominoes"] = dict()
variables["mass_dominoes"]["num_middle_objects"] = [0, 1, 2, 3, 4]


variables["mass_waterpush"] = dict()
variables["mass_waterpush"]["target"] = ["bowl","cone","cube","cylinder","dumbbell","pentagon","pipe","pyramid"]
variables["mass_waterpush"]["tscale"] = ["0.5,0.5,0.5","0.45,0.5,0.45","0.4,0.5,0.4","0.35,0.5,0.35", "0.3,0.5,0.3"]
variables["bouncy_platform"]['use_blocker_with_hole'] = ["0", "1"]
variables["bouncy_platform"]["target"] = ["bowl","cone","cube","cylinder","dumbbell","pentagon","pipe","pyramid"]
variables["bouncy_platform"]["tscale"] = ["0.2,0.2,0.2", "0.3,0.3,0.3", "0.4,0.4,0.4", "0.5,0.5,0.5"]




f = open(template_config, 'r')
template_content = f.read()


var_values =  [*variables[scenario_name].values()]
var_keys =  [*variables[scenario_name].keys()]
print("number of variables", len(var_keys))


def name_generator(vark, varv, cur_var_id, pfx, vars):

    nvar = len(varv)
    for value in varv[cur_var_id]:
        cur_pfx = pfx + "-" + vark[cur_var_id] + "_" + f"{value}"
        cur_list = vars + [(vark[cur_var_id], value)]
        if cur_var_id == nvar - 1: #end
            print(cur_pfx)
            print(cur_list)

            os.mkdir(cur_pfx)
            output_file = os.path.join(cur_pfx, "commandline_args.txt")
            print(output_file)

            fw = open(output_file, 'w')
            for variable, ele in cur_list:
                fw.write(f"--{variable}\n{ele}\n")
            fw.write(template_content)
            fw.close()

        else:
            name_generator(vark, varv, cur_var_id + 1, cur_pfx, cur_list)

name_generator(var_keys, var_values, 0, os.path.join("/".join(template_config.split("/")[:-1]), prefix ), [])


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