

#### mass ###
controller="tdw_physics/target_controllers/dominoes_var.py"
ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
python $controller @$ARGS_PATH/dominoes_pp/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --training_data_mode --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/mass_dominoes_pp/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_curtain  --height 128 --width 128 --num 30


# controller="tdw_physics/target_controllers/waterpush_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/waterpush_pp/pilot_it2_collision_simple_box/commandline_args.txt --training_data_mode --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/mass_waterpush/pilot_it2_collision_simple_box  --height 256 --width 256 --num 40

# controller="tdw_physics/target_controllers/collision_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/collide_pp/pilot_it2_collision_simple_box/commandline_args.txt --training_data_mode  --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/mass_collision/pilot_it2_collision_simple_box5  --height 256 --width 256 --num 20



#### bounciness ###

# no rest
# controller="tdw_physics/target_controllers/bouncy.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/bouncy_pp/pilot_it2_collision_simple_box/commandline_args.txt --training_data_mode --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/bouncy/pilot_it2_collision_simple_box  --height 256 --width 256


# controller="tdw_physics/target_controllers/bouncywall_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/bouncywall_pp/pilot_it2_collision_simple_box/commandline_args.txt --training_data_mode --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/bouncywall/pilot_it2_collision_simple_box  --height 256 --width 256


#### friction ###

# no reset
# controller="tdw_physics/target_controllers/fricramp.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/fricramp_pp/pilot_it2_rollingSliding_simple_ramp_box_2distinct/commandline_args.txt --training_data_mode --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/fricramp/pilot_it2_rollingSliding_simple_ramp_box_2distinct_singleramp5 --height 256 --width 256 --num 20

# no rest -- friction cloth
# controller="tdw_physics/target_controllers/fricrampcloth.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# #python $controller @$ARGS_PATH/fricrampcloth_pp/pilot_it2_rollingSliding_simple_ramp_box_2distinct_singleramp/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/fricrampcloth/pilot_it2_rollingSliding_simple_ramp_box_2distinct_singleramp --height 256 --width 256
# python $controller @$ARGS_PATH/fricrampcloth_pp/pilot_it2_rollingSliding_simple_ramp_box_2distinct/commandline_args.txt --training_data_mode --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/fricrampcloth/pilot_it2_rollingSliding_simple_ramp_box_2distinct_singleramp --height 256 --width 256  --num 30


# controller="tdw_physics/target_controllers/collisionfric_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/collidefric_pp/pilot_it2_collision_simple_box/commandline_args.txt --training_data_mode  --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/collidefric_pp/pilot_it2_collision_simple_box  --height 256 --width 256 --num 30



# rest  rolling


#### viscosity ###
# controller="tdw_physics/target_controllers/fluidhit_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/fluidhit_pp/pilot_it2_drop_simple_box/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/fluidhit_pp2/_pppilot_it2_drop_simple_box --height 512 --width 512
# #python $controller @$ARGS_PATH/fluidhit_pp/pilot_it2_drop_simple_box/commandline_args.txt --dir /home/htung/Downloads/tdw_physics/dump/fluidhit_pp/_pppilot_it2_drop_simple_box --height 256 --width 256


### fluid with slop, no reset
# controller="tdw_physics/target_controllers/fluidslope.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/fluidslope_pp/pilot_it2_drop_simple_box/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/fluidslope_pp/_pppilot_it2_drop_simple_box --height 512 --width 512
#0.40 for sticy object, 0.3 for water


#### cloth ####
# controller="tdw_physics/target_controllers/dropcloth_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/dropcloth_pp/pilot_it2_drop_simple_box/commandline_args.txt --training_data_mode  --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/dropcloth/pilot_it2_drop_simple_box --height 256 --width 256 --num 40

# controller="tdw_physics/target_controllers/clothhit_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/clothhit_pp/pilot_it2_drop_simple_box/commandline_args.txt --training_data_mode --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/clothhit/pilot_it2_drop_simple_box --height 256 --width 256 --num 20

# controller="tdw_physics/target_controllers/clothhang_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/clothhang_pp/pilot_it2_drop_simple_box/commandline_args.txt --training_data_mode  --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/clothhang/pilot_it2_drop_simple_box --height 256 --width 256 --num 20


##bowl,cube,cylinder,octahedron,pentagon,pipe,platonic,pyramid,sphere,torus

#0.5, 0.9, 0

######## not used ########

# controller="tdw_physics/target_controllers/waterhitwall_var.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/droptowater_pp/pilot_it2_drop_simple_box/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/drop2water/pilot_it2_drop_simple_box  --height 512 --width 512





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