
if [ "$HOSTNAME" = aw-m17-R2 ]; then
    printf '%s\n' "alienware laptop"
    PHYSICS_HOME="$HOME"/Documents/2021
    DUMP_DIR="/media/htung/Extreme SSD/fish/tdw_physics/dump"
elif [[ "$HOSTNAME" == *"ccncluster" ]]; then
    printf '%s\n'$HOSTNAME
    PHYSICS_HOME="$HOME"/2021
    DUMP_DIR="/mnt/fs4/hsiaoyut/physion++/mini_v2"
elif [[ "$HOSTNAME" == *"ccncluster" ]]; then
    printf '%s\n'$HOSTNAME
    PHYSICS_HOME="$HOME"/2021
    DUMP_DIR="/mnt/fs4/hsiaoyut/physion++/mini_v2"
else
	printf "unknown hostname"
fi

echo $PHYSICS_HOME
scenario_name="bouncy_platform_pp"
controller="tdw_physics/target_controllers/bouncy.py"
ARGS_PATH=$PHYSICS_HOME"/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"

for args_f in $ARGS_PATH"/"$scenario_name/bouncy_platform*
do
    echo "found file" $args_f
    echo "write data to" "$DUMP_DIR"${args_f#$ARGS_PATH}

    CUDA_VISIBLE_DEVIES=$1 python $controller @$args_f"/commandline_args.txt" --dir "$DUMP_DIR"${args_f#$ARGS_PATH}  --training_data_mode --height 256 --width 256 --gpu $1 --num 10
done

# sname="mass_waterpush"
# scenario_name=$sname"_pp"
# controller="tdw_physics/target_controllers/waterpush_var.py"
# ARGS_PATH=$PHYSICS_HOME"/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"

# for args_f in $ARGS_PATH"/"$scenario_name/$sname-target_*
# do
#     echo "found file" $args_f
#     echo "write data to" "$DUMP_DIR"${args_f#$ARGS_PATH}

#     CUDA_VISIBLE_DEVIES=$1 python $controller @$args_f"/commandline_args.txt" --dir "$DUMP_DIR"${args_f#$ARGS_PATH}  --training_data_mode --height 64 --width 64 --gpu $1 --num 40
# done



# ARGS_PATH=$HOME"/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"



#CUDA_VISIBLE_DEVIES=$1 python $controller @$ARGS_PATH/mass_dominoes_pp/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir "$DUMP_DIR"/mass_dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_curtain  --height 128 --width 128 --gpu 1
