#!/bin/bash

if [$HOSTNAME == "aw-m17-R2"];
then
    STIMULI_ROOT=/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation
    OUTPUT_ROOT=data3
else
    STIMULI_ROOT=../human-physics-benchmarking/stimuli/generation
    OUTPUT_ROOT="D:\hsiaoyut\tdw_physics"

    #STIMULI_ROOT=/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation
    #OUTPUT_ROOT=/mnt/fs0/hsiaoyut/tdw_physics/data
fi


TOWERS_C=tdw_physics/target_controllers/towers.py
NUM=200
#python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb2_fr015_SJ010_mono0_dis0_occ0_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\towers\pilot_towers_nb2_fr015_SJ010_mono0_dis0_occ0_tdwroom" --num $NUM --only_use_flex_objects  --height 64 --width 64 --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
#python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb3_fr015_SJ025_mono1_dis0_occ0_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\towers\pilot_towers_nb3_fr015_SJ025_mono1_dis0_occ0_tdwroom" --num $NUM --only_use_flex_objects  --height 64 --width 64 --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\towers\pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom" --num $NUM --only_use_flex_objects --height 64 --width 64   --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb4_fr015_SJ000_gr015sph_mono1_dis0_occ0_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\towers\pilot_towers_nb4_fr015_SJ000_gr015sph_mono1_dis0_occ0_tdwroom" --num $NUM --only_use_flex_objects --height 64 --width 64   --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb4_fr015_SJ000_gr01_mono1_dis0_occ0_boxroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\towers\pilot_towers_nb4_fr015_SJ000_gr01_mono1_dis0_occ0_boxroom" --num $NUM --only_use_flex_objects --height 64 --width 64   --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb4_SJ025_mono1_dis0_occ0_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\towers\pilot_towers_nb4_SJ025_mono1_dis0_occ0_tdwroom" --num $NUM --only_use_flex_objects  --training_data_mode --height 64 --width 64  --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\towers\pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom" --num $NUM --only_use_flex_objects --height 64 --width 64   --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
