############# dominoes ##################

controller="tdw_physics/target_controllers/dominoes_var.py"
ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
#python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir dump/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_curtain --num 10 --height 128 --width 128
python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2distinct_0middle_tdwroom_fixedcam/commandline_args.txt --dir dump/dominoes/pilot_dominoes_2distinct_0middle_tdwroom_fixedcam_curtain --num 10 --height 128 --width 128
#python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2distinct_3middle_tdwroom_fixedcam/commandline_args.txt --dir dump/dominoes/pilot_dominoes_2distinct_3middle_tdwroom_fixedcam_curtain --num 10 --height 128 --width 128

#python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir /mnt/fs4/hsiaoyut/tdw_fewshot/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam --num 200 --height 128 --width 128


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
