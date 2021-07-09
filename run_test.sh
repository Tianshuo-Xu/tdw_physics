sc_name=clothSagging
python_py=C:/cygwin64/home/hsiaoyut/anaconda/envs/tdw/python


controller_py=C:/cygwin64/home/hsiaoyut/2021/tdw_physics/tdw_physics/target_controllers/soft_push.py
desired_number=2000
group=train
seed=0

echo $total_num
#
#
for entry in "c:/cygwin64/home/hsiaoyut/2021/hum1an-physics-benchmarking/stimuli/generation/pilot-"$sc_name/test1*
do
	num_multiplier=1
    echo $entry
    IFS='/' read -r -a array <<< "$entry"
    arg_name=${array[-1]}
    echo ">>>>>>>>>>>>>>>"
    echo $arg_name
    echo "==============="
    echo $num_multiplier
    $python_py $controller_py @$entry/commandline_args.txt --testing_data_mode --dir "D:/hsiaoyut/tdw_physics/data/"$sc_name/$arg_name/$group --width 1024 --height 1024
    break
done