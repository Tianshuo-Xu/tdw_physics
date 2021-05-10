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
######################################### dominoes ###############################################
#python tdw_physics/target_controllers/dominoes.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-dominoes/pilot_dominoes_2mid_J025R30_tdwroom/commandline_args.txt --dir log2/ --num 2 --height 128 --width 128


#python tdw_physics/target_controllers/dominoes.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-dominoes/pilot_dominoes_4mid_tdwroom/commandline_args.txt --dir log2/ --num 2 --height 512 --width 512

#python tdw_physics/target_controllers/dominoes.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-dominoes/pilot_dominoes_1mid_J025R45_boxroom/commandline_args.txt --dir log2/ --num 2 --height 128 --width 128 --save_meshes

#
DOMINOES_C=tdw_physics/target_controllers/dominoes.py
NUM=1000
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_0mid_d3chairs_o1plants_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes\pilot_dominoes_0mid_d3chairs_o1plants_tdwroom" --num $NUM --only_use_flex_objects  --height 128 --width 128 --save_meshes --port 1701
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_1mid_J025R45_boxroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes\pilot_dominoes_1mid_J025R45_boxroom" --num $NUM  --only_use_flex_objects  --height 128 --width 128 --save_meshes
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_1mid_J025R45_o1flex_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes\pilot_dominoes_1mid_J025R45_o1flex_tdwroom" --num $NUM --only_use_flex_objects --height 128 --width 128 --save_meshes
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_2mid_J020R15_d3chairs_o1plants_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes\pilot_dominoes_2mid_J020R15_d3chairs_o1plants_tdwroom" --num $NUM --only_use_flex_objects --height 128 --width 128 --save_meshes
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_2mid_J025R30_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes\pilot_dominoes_2mid_J025R30_tdwroom" --num $NUM --only_use_flex_objects --height 128 --width 128 --save_meshes
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_4mid_boxroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes\pilot_dominoes_4mid_boxroom"  --num $NUM --only_use_flex_objects --height 128 --width 128 --save_meshes
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_4midRM1_boxroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes\pilot_dominoes_4midRM1_boxroom" --num $NUM --only_use_flex_objects --height 128 --width 128 --save_meshes 
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_4midRM1_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes\pilot_dominoes_4midRM1_tdwroom" --num $NUM --only_use_flex_objects --height 128 --width 128 --save_meshes
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_4mid_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes\pilot_dominoes_4mid_tdwroom" --num $NUM --only_use_flex_objects --height 128 --width 128 --save_meshes
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_default_boxroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes\pilot_dominoes_default_boxroom" --num $NUM --only_use_flex_objects --height 128 --width 128 --save_meshes
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes\pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom" --num $NUM --only_use_flex_objects --height 128 --width 128 --save_meshes


python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\dominoes_test\pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom" --num $NUM --only_use_flex_objects  --save_meshes --write_passes "_img,_id" --random 0 --seed 888 --tcolor None --zcolor None --pcolor None --height 256 --width 256

#export ARGNAME="pilot_dominoes_0mid_d3chairs_o1plants_tdwroom"


#scp *.hdf5 hsiaoyut@node17-ccncluster.stanford.edu:/mnt/fs0/hsiaoyut/tdw_physics/data/pilot_dominoes_4mid_tdwroom
#scp *obj[7-8].obj hsiaoyut@node17-ccncluster.stanford.edu:/mnt/fs0/hsiaoyut/tdw_physics/data/pilot_dominoes_4mid_tdwroom

##################################### linking ####################################################

#python tdw_physics/target_controllers/linking.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-linking/pilot_linking_nl1-5_aNone_bCube_tdwroom/commandline_args.txt --dir data/linking/pilot_linking_nl1-5_aNone_bCube_tdwroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/linking.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-linking/pilot_linking_nl1-6_ms03-7_aCylcap_bCyl_tdwroom/commandline_args.txt --dir data/linking/pilot_linking_nl1-6_ms03-7_aCylcap_bCyl_tdwroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/linking.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-linking/pilot_linking_nl2-3_mg01_aCone_bCyl_boxroom/commandline_args.txt --dir data/linking/pilot_linking_nl2-3_mg01_aCone_bCyl_boxroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/linking.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-linking/pilot_linking_nl5_mg01_aCyl_bCube_boxroom/commandline_args.txt --dir data/linking/pilot_linking_nl5_mg01_aCyl_bCube_boxroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/linking.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-linking/pilot_linking_nl1-8_aCyl_bCube_tdwroom/commandline_args.txt --dir data/linking/pilot_linking_nl1-8_aCyl_bCube_tdwroom  --save_meshes --num 10 --height 128 --width 128

##################################### towers ####################################################

#python tdw_physics/target_controllers/towers.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-towers/pilot_towers_nb2_fr015_SJ010_mono0_tdwroom/commandline_args.txt --dir data/towers/pilot_towers_nb2_fr015_SJ010_mono0_tdwroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/towers.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-towers/pilot_towers_nb3_fr015_SJ025_mono1_tdwroom/commandline_args.txt --dir data/towers/pilot_towers_nb3_fr015_SJ025_mono1_tdwroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/towers.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-towers/pilot_towers_nb5_fr015_SJ030_mono0_boxroom/commandline_args.txt --dir data/towers/pilot_towers_nb5_fr015_SJ030_mono0_boxroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/towers.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-towers/pilot_towers_nb4_fr015_SJ000_gr015sph_mono1_tdwroom/commandline_args.txt --dir data/towers/pilot_towers_nb4_fr015_SJ000_gr015sph_mono1_tdwroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/towers.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-towers/pilot_towers_nb4_fr015_SJ000_gr-01_mono0_tdwroom/commandline_args.txt --dir data/towers/pilot_towers_nb4_fr015_SJ000_gr-01_mono0_tdwroom --save_meshes --num 10 --height 128 --width 128


##################################### collision ####################################################

#python tdw_physics/target_controllers/collision.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-collision/iteration-1/pilot_it1_collision_zone_jitter_box/commandline_args.txt --dir data/collision/pilot_it1_collision_zone_jitter_box  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/collision.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-collision/iteration-1/pilot_it1_collision_yeet_box/commandline_args.txt --dir data/collision/pilot_it1_collision_yeet_box  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/collision.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-collision/iteration-1/pilot_it1_collision_slow_box/commandline_args.txt --dir data/collision/pilot_it1_collision_slow_box  --save_meshes --num 10 --height 128 --width 128


##################################### drop ####################################################

#python tdw_physics/target_controllers/drop.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-drop/iteration-1/pilot_it1_drop_all_bowls_box/commandline_args.txt --dir data/drop/pilot_it1_drop_all_bowls_box --save_meshes  --temp "D:/temp.hdf5"  --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/drop.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-drop/iteration-1/pilot_it1_drop_sidezone_box/commandline_args.txt --dir data/drop/pilot_it1_drop_sidezone_box  --save_meshes --temp "D:/temp.hdf5" --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/drop.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-drop/iteration-1/pilot_it1_drop_simple_box/commandline_args.txt --dir data/drop/pilot_it1_drop_simple_box --save_meshes --temp "D:/temp.hdf5" --num 10 --height 128 --width 128



##################################### containment ####################################################

#python tdw_physics/target_controllers/containment.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-containment/pilot-containment-bowl/commandline_args.txt --dir data/containment/pilot-containment-bowl --save_meshes  --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/containment.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-containment/pilot-containment-multi/commandline_args.txt --dir data/containment/pilot-containment-multi  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/containment.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-containment/pilot-containment-torus/commandline_args.txt --dir data/containment/pilot-containment-torus --save_meshes --num 10 --height 128 --width 128


##################################### rollingSliding ####################################################

#python tdw_physics/target_controllers/rolling_sliding.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-rollingSliding/iteration-1/pilot_it1_rollingSliding_simple_left/commandline_args.txt --dir data/rollingSliding/pilot_it1_rollingSliding_simple_left --save_meshes  --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/rolling_sliding.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-rollingSliding/iteration-1/pilot-containment-multi/commandline_args.txt --dir data/rollingSliding/pilot-containment-multi  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/rolling_sliding.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-rollingSliding/iteration-1/pilot-containment-torus/commandline_args.txt --dir data/rollingSliding/pilot-containment-torus --save_meshes --num 10 --height 128 --width 128





#python tdw_physics/target_controllers/dominoes.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-dominoes/pilot_dominoes_4mid_boxroom/commandline_args.txt --dir data/dominoes/pilot_dominoes_4mid_boxroom --num 100 --height 128 --width 128 --mscale "0.1,0.5,0.25"  --save_meshes

# linking # weird stacking of objects
#python tdw_physics/target_controllers/linking.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-linking/pilot_linking_nl1-5_aNone_bCube_tdwroom/commandline_args.txt --dir log2/ --num 2 --height 512 --width 512 --mscale "0.1,0.5,0.25"  --save_meshes


# tower
#python tdw_physics/target_controllers/towers.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-towers/pilot_towers_nb2_fr015_SJ010_mono0_tdwroom/commandline_args.txt --dir log/towers --num 2 --height 512 --width 512 --mscale "0.1,0.5,0.25"  --save_meshes


# liquid

#python tdw_physics/target_controllers/dominoes.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-dominoes/pilot_dominoes_1mid_J025R45_boxroom/commandline_args.txt --dir log2/ --num 2 --height 128 --width 128 --mscale "0.1,0.5,0.25"  --save_meshes


# clothes

#python tdw_physics/target_controllers/dominoes.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-dominoes/pilot_dominoes_1mid_J025R45_boxroom/commandline_args.txt --dir log2/ --num 2 --height 128 --width 128 --mscale "0.1,0.5,0.25"  --save_meshes
