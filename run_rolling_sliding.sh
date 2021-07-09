#!/bin/bash
#
#total_num=277 #dominoes # not finlize, don't use
#total_num=240 # collisions, equally split across the args
total_num=182 # rollingSliding, equally split across the args
#total_num=200 # droping, equally split across the args
sc_name=rollingSliding
python_py=C:/cygwin64/home/hsiaoyut/anaconda/envs/tdw/python


# div ()  # Arguments: dividend and divisor
# {
#         if [ $2 -eq 0 ]; then echo division by 0; exit; fi
#         local p=12                            # precision
#         local c=${c:-0}                       # precision counter
#         local d=.                             # decimal separator
#         local r=$(($1/$2)); echo -n $r        # result of division
#         local m=$(($r*$2))
#         [ $c -eq 0 ] && [ $m -ne $1 ] && echo -n $d
#         [ $1 -eq $m ] || [ $c -eq $p ] && echo && return
#         local e=$(($1-$m))
#         c=$(($c+1))
#         div $(($e*10)) $2
# }  

controller_rollingsliding_py=C:/cygwin64/home/hsiaoyut/2021/tdw_physics/tdw_physics/target_controllers/rolling_sliding.py
controller_collision_py=C:/cygwin64/home/hsiaoyut/2021/tdw_physics/tdw_physics/target_controllers/collision.py
desired_number=2000
group=train
seed=0

echo $total_num
#
#
# for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/pilot_it2_rollingSliding_simple_collision_box_1_dis_1_occ*
# do
# 	num_multiplier=11
#     echo $entry
#     IFS='/' read -r -a array <<< "$entry"
#     arg_name=${array[-1]}
#     echo ">>>>>>>>>>>>>>>"
#     echo $arg_name
#     echo "==============="
#     echo $num_multiplier
#     $python_py $controller_collision_py @$entry/commandline_args.txt --training_data_mode --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --height 256 --width 256 --seed $seed --num_multiplier $num_multiplier --save_passes "" --write_passes "_img,_id"  --save_meshes &
# done

# for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/pilot_it2_rollingSliding_simple_ledge_box_1_dis_1_occ*
# do
# 	num_multiplier=11
#     echo $entry
#     IFS='/' read -r -a array <<< "$entry"
#     arg_name=${array[-1]}
#     echo ">>>>>>>>>>>>>>>"
#     echo $arg_name
#     echo "==============="
#     echo $num_multiplier
#     $python_py $controller_rollingsliding_py @$entry/commandline_args.txt --training_data_mode --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --height 256 --width 256 --seed $seed --num_multiplier $num_multiplier --save_passes "" --write_passes "_img,_id"  --save_meshes &
# done

desired_number=200
group=valid
seed=1


# for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/*collision*
# do
# 	num_multiplier=1.1
#     echo $entry
#     IFS='/' read -r -a array <<< "$entry"
#     arg_name=${array[-1]}
#     echo ">>>>>>>>>>>>>>>"
#     echo $arg_name
#     echo "==============="
#     echo $num_multiplier
#     $python_py $controller_collision_py @$entry/commandline_args.txt --training_data_mode --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --height 256 --width 256 --seed $seed --num_multiplier $num_multiplier --save_passes "" --write_passes "_img,_id"  --save_meshes &
# done

# for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/*ledge*
# do
# 	num_multiplier=1.1
#     echo $entry
#     IFS='/' read -r -a array <<< "$entry"
#     arg_name=${array[-1]}
#     echo ">>>>>>>>>>>>>>>"
#     echo $arg_name
#     echo "==============="
#     echo $num_multiplier
#     $python_py $controller_rollingsliding_py @$entry/commandline_args.txt --training_data_mode --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --height 256 --width 256 --seed $seed --num_multiplier $num_multiplier --save_passes "" --write_passes "_img,_id"  --save_meshes &
# done

#
#
# for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/*ramp*
# do
# 	num_multiplier=1.1
#     echo $entry
#     IFS='/' read -r -a array <<< "$entry"
#     arg_name=${array[-1]}
#     echo ">>>>>>>>>>>>>>>"
#     echo $arg_name
#     echo "==============="
#     echo $num_multiplier
#     $python_py $controller_rollingsliding_py @$entry/commandline_args.txt --training_data_mode --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --height 256 --width 256 --seed $seed --num_multiplier $num_multiplier --save_passes "" --write_passes "_img,_id"  --save_meshes &
# done

desired_number=1000
group=train_readout
seed=2



# for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/pilot_it2_rollingSliding_simple_collision_tdw_2_dis_2_occ*
# do
# 	num_multiplier=5.55
#     echo $entry
#     IFS='/' read -r -a array <<< "$entry"
#     arg_name=${array[-1]}
#     echo ">>>>>>>>>>>>>>>"
#     echo $arg_name
#     echo "==============="
#     echo $num_multiplier
#     $python_py $controller_collision_py @$entry/commandline_args.txt --readout_data_mode  --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --height 256 --width 256 --seed $seed --num_multiplier $num_multiplier --save_passes "" --write_passes "_img,_id"  --save_meshes &
# done

# for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/*ledge*
# do
# 	num_multiplier=5.5
#     echo $entry
#     IFS='/' read -r -a array <<< "$entry"
#     arg_name=${array[-1]}
#     echo ">>>>>>>>>>>>>>>"
#     echo $arg_name
#     echo "==============="
#     echo $num_multiplier
#     $python_py $controller_rollingsliding_py @$entry/commandline_args.txt --readout_data_mode  --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --height 256 --width 256 --seed $seed --num_multiplier $num_multiplier --save_passes "" --write_passes "_img,_id"  --save_meshes &
# done

# for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/*ramp*
# do
# 	num_multiplier=5.5
#     echo $entry
#     IFS='/' read -r -a array <<< "$entry"
#     arg_name=${array[-1]}
#     echo ">>>>>>>>>>>>>>>"
#     echo $arg_name
#     echo "==============="
#     echo $num_multiplier
#     $python_py $controller_rollingsliding_py @$entry/commandline_args.txt --readout_data_mode  --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --height 256 --width 256 --seed $seed --num_multiplier $num_multiplier --save_passes "" --write_passes "_img,_id"  --save_meshes &
# done



desired_number=100
group=valid_readout
seed=3
# #
# #

# for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/*collision*
# do
# 	num_multiplier=0.55
#     echo $entry
#     IFS='/' read -r -a array <<< "$entry"
#     arg_name=${array[-1]}
#     echo ">>>>>>>>>>>>>>>"
#     echo $arg_name
#     echo "==============="
#     echo $num_multiplier
#     $python_py $controller_collision_py @$entry/commandline_args.txt --readout_data_mode  --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --height 256 --width 256 --seed $seed --num_multiplier $num_multiplier --save_passes "" --write_passes "_img,_id"  --save_meshes &
# done

# wait

# for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/*ledge*
# do
# 	num_multiplier=0.55
#     echo $entry
#     IFS='/' read -r -a array <<< "$entry"
#     arg_name=${array[-1]}
#     echo ">>>>>>>>>>>>>>>"
#     echo $arg_name
#     echo "==============="
#     echo $num_multiplier
#     $python_py $controller_rollingsliding_py @$entry/commandline_args.txt --readout_data_mode  --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --height 256 --width 256 --seed $seed --num_multiplier $num_multiplier --save_passes "" --write_passes "_img,_id"  --save_meshes &
# done

for entry in "c:/cygwin64/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation/pilot-"$sc_name/*ramp*
do
	num_multiplier=0.55
    echo $entry
    IFS='/' read -r -a array <<< "$entry"
    arg_name=${array[-1]}
    echo ">>>>>>>>>>>>>>>"
    echo $arg_name
    echo "==============="
    echo $num_multiplier
    $python_py $controller_rollingsliding_py @$entry/commandline_args.txt --readout_data_mode  --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --height 256 --width 256 --seed $seed --num_multiplier $num_multiplier --save_passes "" --write_passes "_img,_id"  --save_meshes &
done
