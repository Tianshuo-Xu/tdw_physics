#count file
import os

cat = "mass"
folder = "/mnt/fs4/hsiaoyut/physion++/data_v1"

#mass_collision_pp 287
#mass_waterpush_pp 663
#mass_dominoes_pp 637
#total_count 1587

#friction_cloth_pp 659
#friction_collision_pp 666
#friction_platform_pp 672
#total_count 1997

#bouncy_platform_pp 995
#bouncy_wall_pp 995
#total_count 1990


count = 0
for scene_name in os.listdir(folder):
    if scene_name.startswith(cat):
        arg_count = 0
        for arg_name in os.listdir(os.path.join(folder, scene_name)):
            pkl_files = [f for f in os.listdir(os.path.join(folder, scene_name, arg_name)) if f.endswith(".pkl")]
            arg_count += len(pkl_files)
            #print("    ", arg_name, len(pkl_files))
        count += arg_count
        print(scene_name, arg_count)

print("total_count", count)