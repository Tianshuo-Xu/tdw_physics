#generate config combination
import os

template_config = "/home/htung/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs/mass_dominoes_pp/commandline_args_template.txt"
prefix = "mass_dominoes_"
variables = dict()
variables["num_middle_objects"] = [0, 1, 2, 3, 4]

f = open(template_config, 'r')
content = f.read()


# modify to dfs
for variable, values in variables.items():
    for ele in values:
        output_dir = os.path.join("/".join(template_config.split("/")[:-1]), prefix + f"{variable}_{ele}")
        os.mkdir(output_dir)
        output_file = os.path.join(output_dir, "commandline_args.txt")
        print(output_file)

        fw = open(output_file, 'w')
        fw.write(f"--{variable}\n{ele}\n" + content)
        fw.close()

import ipdb; ipdb.set_trace()

print("hello")