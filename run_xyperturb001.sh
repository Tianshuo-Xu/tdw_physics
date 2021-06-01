#!/bin/bash
#
total_num=277 #dominoes # not finlize, don't use
#total_num=240 # collisions, equally split across the args
#total_num=220 #towers
#total_num=175 #linking, containment

#total_num=182 # rollingSliding, equally split across the args
#total_num=200 # droping, equally split across the args
#dominoes
sc_name=$1
python_py=C:/cygwin64/home/hsiaoyut/anaconda/envs/tdw/python

controller_py="C:/cygwin64/home/hsiaoyut/2021/tdw_physics/tdw_physics/target_controllers/$sc_name.py"

echo $sc_name
echo $controller_py

string_match="pilot"
if [ "$sc_name" == "clothSagging" ]; then
    string_match=""
fi

for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/"$string_match"*
do

	num_multiplier=1
    echo $entry

    IFS='/' read -r -a array <<< "$entry"
    arg_name=${array[-1]}
    echo ">>>>>>>>>>>>>>>"
    echo $arg_name
    echo "==============="
	if [ "$sc_name" == "rollingSliding" ]; then
		if [[ "$arg_name" == *"collision"* ]]; then
			#echo "hello"
			controller_py="C:/cygwin64/home/hsiaoyut/2021/tdw_physics/tdw_physics/target_controllers/collision.py"
        else
            controller_py="C:/cygwin64/home/hsiaoyut/2021/tdw_physics/tdw_physics/target_controllers/$sc_name.py"
        fi
        echo $controller_py
    elif [ "$sc_name" == "clothSagging" ]; then
    	controller_py="C:/cygwin64/home/hsiaoyut/2021/tdw_physics/tdw_physics/target_controllers/cloth_sagging.py"
    	echo $controller_py
	fi
    #echo @$entry/commandline_args.txt
    echo $num_multiplier
    $python_py $controller_py @$entry/commandline_args.txt --testing_noimg_data_mode --is_perturb --height 64 --width 64  --dir "D:/hsiaoyut/tdw_physics/pe_data/xy_perturb0.01/"$sc_name/$arg_name/$group
done



# for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/pilot_*[1]mid_*
# do
# 	num_multiplier=1
#     echo $entry
#     IFS='/' read -r -a array <<< "$entry"
#     arg_name=${array[-1]}
#     echo ">>>>>>>>>>>>>>>"
#     echo $arg_name
#     echo "==============="
#     echo $num_multiplier
#     $python_py $controller_py @$entry/commandline_args.txt --testing_data_mode --dir "D:/hsiaoyut/tdw_physics/pe_data/xy_perturb_with_images_det6/"$sc_name/$arg_name/$group --height 64 --width 64   --save_passes "" --write_passes "_img,_id"
# done
