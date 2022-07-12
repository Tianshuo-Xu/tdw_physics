### scripts for running fluid and cloth simulation with obi simulation
### You can simply genreate the config file by copying those files from physion-drop. they basically share the same params



### controllers with xxx_var.py is intended for generating multi-cut video with curtain-reset.
### if you don't want the reset, you can start from standard dominoes controller and copy paste obi-relevant lines:
### (please search for "obi" in the controller to see the main change. The changes should be around 5 lines of code)


controller="tdw_physics/target_controllers/fluid_drop_var.py"
ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021/stimuli/generation/configs"
#python $controller @$ARGS_PATH/physionpp-dropfluid/pilot_it2_drop_all_bowls_box/commandline_args.txt --port 1071 --dir /YOURDIR/dropfluid/pilot_it2_drop_all_bowls_box  --height 128 --width 128


controller="tdw_physics/target_controllers/cloth_drop_var.py"
ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021/stimuli/generation/configs"
#python $controller @$ARGS_PATH/physionpp-dropfluid/pilot_it2_drop_all_bowls_box/commandline_args.txt --port 1071 --dir /YOURDIR/dropcloth/pilot_it2_drop_all_bowls_box  --height 128 --width 128
