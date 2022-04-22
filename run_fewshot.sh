############# dominoes ##################

controller="tdw_physics/target_controllers/dominoes_var.py"
ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"

#python $controller @$ARGS_PATH/dominoes_pp/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/dominoes2/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_curtain  --height 128 --width 128
#python $controller @$ARGS_PATH/dominoes_pp/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_curtain  --height 256 --width 256
# python $controller @$ARGS_PATH/dominoes_pp/pilot_dominoes_2distinct_0middle_tdwroom_fixedcam/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/dominoes/pilot_dominoes_2distinct_0middle_tdwroom_fixedcam_curtain  --height 256 --width 256
# python $controller @$ARGS_PATH/dominoes_pp/pilot_dominoes_2distinct_2middle_tdwroom_fixedcam/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/dominoes/pilot_dominoes_2distinct_2middle_tdwroom_fixedcam_curtain  --height 256 --width 256
# python $controller @$ARGS_PATH/dominoes_pp/pilot_dominoes_2distinct_3middle_tdwroom_fixedcam/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/dominoes/pilot_dominoes_2distinct_3middle_tdwroom_fixedcam_curtain  --height 256 --width 256
# python $controller @$ARGS_PATH/dominoes_pp/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_familiarization/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_familiarization --height 256 --width 256
# python $controller @$ARGS_PATH/dominoes_pp/pilot_dominoes_2distinct_3middleRM1_tdwroom_fixedcam/commandline_args.txt --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/dominoes/pilot_dominoes_2distinct_3middleRM1_tdwroom_fixedcam --testing_data_mode --height 256 --width 256


############fluid ##########################

controller="tdw_physics/target_controllers/fluid_drop_var.py"
ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"

python $controller @$ARGS_PATH/physionpp-dropfluid/pilot_it2_drop_all_bowls_box/commandline_args.txt --port 1071 --dir /media/htung/Extreme\ SSD/fish/tdw_physics/dump/drop/pilot_it2_drop_all_bowls_box  --height 128 --width 128

#python tdw_physics/target_controllers/drop.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-drop/iteration-1/pilot_it1_drop_all_bowls_box/commandline_args.txt --dir data/drop/pilot_it1_drop_all_bowls_box --save_meshes  --temp "D:/temp.hdf5"  --num 10 --height 128 --width 128

# try to add a robot
# controller="tdw_physics/target_controllers/dominoes_var_continue.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir dump/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_conti --num 200 --height 128 --width 128

############# sliding ##################

controller="tdw_physics/target_controllers/rolling_sliding_var.py"
ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
#python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir dump/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam --num 200 --height 128 --width 128

#python $controller @$ARGS_PATH/roll/pilot_it2_rollingSliding_simple_ramp_box_2distinct/commandline_args.txt --dir dump/roll/pilot_it2_rollingSliding_simple_ramp_box_2distinct --num 200 --height 128 --width 128


######### clothSagging ###############################

controller="tdw_physics/target_controllers/cloth_sagging.py"
ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
#python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir dump/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam --num 200 --height 128 --width 128

#python $controller @$ARGS_PATH/drape/test11/commandline_args.txt --dir dump/drape/test11 --num 200 --height 128 --width 128
