

#### mass ###
# controller="tdw_physics/target_controllers/dominoes_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/dominoes_pp/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/dominoes2/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_curtain  --height 128 --width 128


# controller="tdw_physics/target_controllers/waterpush_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/waterpush_pp/pilot_it2_collision_simple_box/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/waterpush/pilot_it2_collision_simple_box  --height 256 --width 256

# controller="tdw_physics/target_controllers/collision_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/collide_pp/pilot_it2_collision_simple_box/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/dominoes2/pilot_it2_collision_simple_box  --height 128 --width 128



#### bounciness ###

# controller="tdw_physics/target_controllers/bouncy.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/bouncy_pp/pilot_it2_collision_simple_box/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/bouncy/pilot_it2_collision_simple_box  --height 256 --width 256


controller="tdw_physics/target_controllers/bouncywall_var.py"
ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
python $controller @$ARGS_PATH/bouncywall_pp/pilot_it2_collision_simple_box/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/bouncywall/pilot_it2_collision_simple_box  --height 256 --width 256








######## not used ########

# controller="tdw_physics/target_controllers/masspush_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/masspush_pp/pilot_it2_collision_simple_box/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/masspush/pilot_it2_masspush_simple_box  --height 128 --width 128


### bad
# controller="tdw_physics/target_controllers/drop2water_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/droptowater_pp/pilot_it2_drop_simple_box/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/drop2water/pilot_it2_drop_simple_box  --height 512 --width 512


# controller="tdw_physics/target_controllers/support_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/dominoes_pp/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/dominoes2/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_curtain  --height 128 --width 128