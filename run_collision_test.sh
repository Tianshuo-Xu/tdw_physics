#!/bin/bash
#
total_num=277 #dominoes # not finlize, don't use
#total_num=240 # collisions, equally split across the args
#total_num=220 #towers
#total_num=175 #linking, containment

#total_num=182 # rollingSliding, equally split across the args
#total_num=200 # droping, equally split across the args
sc_name=dominoes
python_py=C:/cygwin64/home/hsiaoyut/anaconda/envs/tdw/python

controller_py=C:/cygwin64/home/hsiaoyut/2021/tdw_physics/tdw_physics/target_controllers/dominoes.py




for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/pilot_*0mid_
do
	num_multiplier=1
    echo $entry
    IFS='/' read -r -a array <<< "$entry"
    arg_name=${array[-1]}
    echo ">>>>>>>>>>>>>>>"
    echo $arg_name
    echo "==============="
    echo $num_multiplier
    $python_py $controller_py @$entry/commandline_args.txt --testing_noimg_data_mode --is_purturb --dir "D:/hsiaoyut/tdw_physics/pe_data/xy_purturb5/"$sc_name/$arg_name/$group   --save_passes "" --write_passes ",_id"
done
