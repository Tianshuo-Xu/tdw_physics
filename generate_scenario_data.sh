#!/bin/bash

controller="./tdw_physics/target_controllers/$1.py"
stim_dirs=`ls $2 | grep pilot`
# stim_dirs=`ls $2 | grep ramp`
out_dir=$3
gpu=$4

if [[ "$#" = "5" ]]; then
    num="--num $5";
else
    num=""
fi

echo $stim_dirs
echo $controller
echo $num

for dir in $stim_dirs
do
    echo $dir;
    config="$2/$dir/commandline_args.txt";
    newdir="$dir-redyellow"
    cmd="python $controller @$config --testing_data_mode --dir $out_dir/$newdir --gpu $gpu $num";
    echo $cmd;
    $cmd;
    echo "completeted regeneration of $num $dir stims";
done
