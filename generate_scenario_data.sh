#!/bin/bash

controller="./tdw_physics/target_controllers/$1.py"
stim_dirs=`ls $2`
out_dir=$3

if [[ "$#" = "4" ]]; then
    num="--num $4";
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
    cmd="python $controller @$config --dir $out_dir/$dir $num";
    echo $cmd;
    $cmd;
    echo "completeted regeneration of $num $dir stims"; 
done
