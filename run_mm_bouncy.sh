
if [ "$HOSTNAME" = aw-m17-R2 ]; then
    printf '%s\n' "alienware laptop"
    PHYSICS_HOME="$HOME"/Documents/2021
    DUMP_DIR="/media/htung/Extreme SSD/fish/tdw_physics/dump_mini"
elif [[ "$HOSTNAME" == *"ccncluster" ]]; then
    printf '%s\n'$HOSTNAME
    PHYSICS_HOME="$HOME"/2021
    DUMP_DIR="/mnt/fs4/hsiaoyut/physion++/mini_v3"
elif [[ "$HOSTNAME" == *"ccncluster" ]]; then
    printf '%s\n'$HOSTNAME
    PHYSICS_HOME="$HOME"/2021
    DUMP_DIR="/mnt/fs4/hsiaoyut/physion++/mini_v3"
else
	printf "unknown hostname"
fi

echo $PHYSICS_HOME
scenario_name="bouncy_platform_pp"
controller="tdw_physics/target_controllers/bouncy.py"
ARGS_PATH=$PHYSICS_HOME"/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"


counter=0
for args_f in $ARGS_PATH"/"$scenario_name/bouncy_platform*
do
    echo "found file" $args_f
    echo "write data to" "$DUMP_DIR"${args_f#$ARGS_PATH}
    $((counter++))

    CUDA_VISIBLE_DEVIES=$1 python $controller @$args_f"/commandline_args.txt" --dir "$DUMP_DIR"${args_f#$ARGS_PATH}  --training_data_mode --height 256 --width 256 --gpu $1 --num 5 --seed $counter
done


echo $PHYSICS_HOME
scenario_name="bouncy_wall_pp"
controller="tdw_physics/target_controllers/bouncywall_var.py"
ARGS_PATH=$PHYSICS_HOME"/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
counter=0
for args_f in $ARGS_PATH"/"$scenario_name/bouncy_wall*
do
    echo "found file" $args_f
    echo "write data to" "$DUMP_DIR"${args_f#$ARGS_PATH}
    $((counter++))

    CUDA_VISIBLE_DEVIES=$1 python $controller @$args_f"/commandline_args.txt" --dir "$DUMP_DIR"${args_f#$ARGS_PATH}  --training_data_mode --height 256 --width 256 --gpu $1 --num 5  --seed $counter
done
