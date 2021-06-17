#!/bin/bash

# if [$HOSTNAME == "aw-m17-R2"];
# then
STIMULI_ROOT=/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation
OUTPUT_ROOT="/media/htung/Extreme SSD/fish/tdw_physics/data"
# else
#     STIMULI_ROOT=/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation
#     OUTPUT_ROOT=/cygdrive/d/hsiaoyut/tdw_physics/data

#     # STIMULI_ROOT=/home/hsiaoyut/2021/human-physics-benchmarking/stimuli/generation
#     # OUTPUT_ROOT=/mnt/fs0/hsiaoyut/tdw_physics/data
# fi
######################################### dominoes ###############################################
#python tdw_physics/target_controllers/dominoes.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-dominoes/pilot_dominoes_2mid_J025R30_tdwroom/commandline_args.txt --dir log2/ --num 2 --height 128 --width 128


#python tdw_physics/target_controllers/dominoes.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-dominoes/pilot_dominoes_4mid_tdwroom/commandline_args.txt --dir log2/ --num 2 --height 512 --width 512

#python tdw_physics/target_controllers/dominoes.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-dominoes/pilot_dominoes_1mid_J025R45_boxroom/commandline_args.txt --dir log2/ --num 2 --height 128 --width 128 --save_meshes

#
DOMINOES_C=tdw_physics/target_controllers/dominoes.py
NUM=1000
# python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_0mid_d3chairs_o1plants_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/dominoes/pilot_dominoes_0mid_d3chairs_o1plants_tdwroom" --num $NUM --only_use_flex_objects  --height 128 --width 128 --save_meshes
# python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_1mid_J025R45_boxroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/dominoes/pilot_dominoes_1mid_J025R45_boxroom" --num $NUM   --only_use_flex_objects  --height 128 --width 128 --save_meshes
# python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_1mid_J025R45_o1flex_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/dominoes/pilot_dominoes_1mid_J025R45_o1flex_tdwroom" --num $NUM  --only_use_flex_objects --height 128 --width 128 --save_meshes
# python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_2mid_J020R15_d3chairs_o1plants_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/dominoes/pilot_dominoes_2mid_J020R15_d3chairs_o1plants_tdwroom" --num $NUM  --only_use_flex_objects --height 128 --width 128 --save_meshes
# python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_2mid_J025R30_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/dominoes/pilot_dominoes_2mid_J025R30_tdwroom" --num $NUM  --only_use_flex_objects --height 128 --width 128 --save_meshes
# python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_4mid_boxroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/dominoes/pilot_dominoes_4mid_boxroom"  --num $NUM  --only_use_flex_objects --height 128 --width 128 --save_meshes
# python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_4midRM1_boxroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/dominoes/pilot_dominoes_4midRM1_boxroom" --num $NUM  --only_use_flex_objects --height 128 --width 128 --save_meshes
# python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_4midRM1_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/dominoes/pilot_dominoes_4midRM1_tdwroom" --num $NUM  --only_use_flex_objects --height 128 --width 128 --save_meshes
# python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_4mid_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/dominoes/pilot_dominoes_4mid_tdwroom" --num $NUM  --only_use_flex_objects --height 128 --width 128 --save_meshes
# python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_default_boxroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/dominoes/pilot_dominoes_default_boxroom" --num $NUM  --only_use_flex_objects --height 128 --width 128 --save_meshes
#python $DOMINOES_C @$STIMULI_ROOT/pilot-dominoes/pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom/commandline_args.txt --dir "data/dominoes/pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom_debug" --num $NUM  --only_use_flex_objects --height 128 --width 128 --save_meshes


######################################### slidingRolling ###############################################

ROLLINGSLIDING_C=tdw_physics/target_controllers/rolling_sliding.py
NUM=500
#python $ROLLINGSLIDING_C @$STIMULI_ROOT/pilot-rollingSliding/pilot_it2_rollingSliding_simple_collision_box/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/rollingSliding/pilot_it2_rollingSliding_simple_collision_box" --num $NUM --only_use_flex_objects  --height 64 --width 64 --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes --port 1071
#python $ROLLINGSLIDING_C @$STIMULI_ROOT/pilot-rollingSliding/pilot_it2_rollingSliding_simple_ledge_box/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/rollingSliding/pilot_it2_rollingSliding_simple_ledge_box" --num $NUM --only_use_flex_objects  --height 64 --width 64 --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes --port 1071
#python $ROLLINGSLIDING_C @$STIMULI_ROOT/pilot-rollingSliding/pilot_it2_rollingSliding_simple_ramp_box/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/rollingSliding/pilot_it2_rollingSliding_simple_ramp_box" --num $NUM --only_use_flex_objects  --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes --port 1071




##################################### linking ####################################################


LINKING_C=tdw_physics/target_controllers/linking.py
NUM=500
# python $LINKING_C @$STIMULI_ROOT/pilot-linking/pilot_linking_nl1-5_aNone_bCube_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/rollingSliding/pilot_linking_nl1-5_aNone_bCube_tdwroom" --num $NUM --only_use_flex_objects  --height 64 --width 64 --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
# python $LINKING_C @$STIMULI_ROOT/pilot-linking/pilot_linking_nl2-3_mg01_aCone_bCyl_boxroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/rollingSliding/pilot_linking_nl2-3_mg01_aCone_bCyl_boxroom" --num $NUM --only_use_flex_objects  --height 64 --width 64 --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
# python $LINKING_C @$STIMULI_ROOT/pilot-linking/pilot_linking_nl4-8_mg-005_aCyl_bCyl_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/rollingSliding/pilot_linking_nl4-8_mg-005_aCyl_bCyl_tdwroom" --num $NUM --only_use_flex_objects  --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes






#python tdw_physics/target_controllers/linking.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-linking/pilot_linking_nl1-5_aNone_bCube_tdwroom/commandline_args.txt --dir data/linking/pilot_linking_nl1-5_aNone_bCube_tdwroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/linking.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-linking/pilot_linking_nl1-6_ms03-7_aCylcap_bCyl_tdwroom/commandline_args.txt --dir data/linking/pilot_linking_nl1-6_ms03-7_aCylcap_bCyl_tdwroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/linking.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-linking/pilot_linking_nl2-3_mg01_aCone_bCyl_boxroom/commandline_args.txt --dir data/linking/pilot_linking_nl2-3_mg01_aCone_bCyl_boxroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/linking.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-linking/pilot_linking_nl5_mg01_aCyl_bCube_boxroom/commandline_args.txt --dir data/linking/pilot_linking_nl5_mg01_aCyl_bCube_boxroom  --save_meshes --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/linking.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-linking/pilot_linking_nl1-8_aCyl_bCube_tdwroom/commandline_args.txt --dir data/linking/pilot_linking_nl1-8_aCyl_bCube_tdwroom  --save_meshes --num 10 --height 128 --width 128

##################################### towers ####################################################



TOWERS_C=tdw_physics/target_controllers/towers.py
NUM=200
# python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb2_fr015_SJ010_mono0_dis0_occ0_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/towers/pilot_towers_nb2_fr015_SJ010_mono0_dis0_occ0_tdwroom" --num $NUM --only_use_flex_objects  --height 64 --width 64 --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
# python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb3_fr015_SJ025_mono1_dis0_occ0_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/towers/pilot_towers_nb3_fr015_SJ025_mono1_dis0_occ0_tdwroom" --num $NUM --only_use_flex_objects  --height 64 --width 64 --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
# python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/towers/pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom" --num $NUM --only_use_flex_objects  --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
# python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb4_fr015_SJ000_gr015sph_mono1_dis0_occ0_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/towers/pilot_towers_nb4_fr015_SJ000_gr015sph_mono1_dis0_occ0_tdwroom" --num $NUM --only_use_flex_objects  --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
# python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb4_fr015_SJ000_gr01_mono1_dis0_occ0_boxroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/towers/pilot_towers_nb4_fr015_SJ000_gr01_mono1_dis0_occ0_boxroom" --num $NUM --only_use_flex_objects  --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
# python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb4_SJ025_mono1_dis0_occ0_tdwroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/towers/pilot_towers_nb4_SJ025_mono1_dis0_occ0_tdwroom" --num $NUM --only_use_flex_objects  --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
# python $TOWERS_C @$STIMULI_ROOT/pilot-towers/pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom/commandline_args.txt --dir "/media/htung/Extreme SSD/fish/tdw_physics/data/towers/pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom" --num $NUM --only_use_flex_objects  --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes




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

DROP_C=tdw_physics/target_controllers/drop.py

NUM=10
# python $DROP_C @$STIMULI_ROOT/pilot-drop/pilot_it2_drop_all_bowls_box/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\drop\pilot_it2_drop_all_bowls_box" --num $NUM --only_use_flex_objects  --height 64 --width 64 --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
# python $DROP_C@$STIMULI_ROOT/pilot-drop/pilot_it2_drop_sidezone_box/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\drop\pilot_it2_drop_sidezone_box" --num $NUM --only_use_flex_objects  --height 64 --width 64 --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
# python $DROP_C @$STIMULI_ROOT/pilot-drop/pilot_it2_drop_simple_box/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\drop\pilot_it2_drop_simple_box" --num $NUM --only_use_flex_objects  --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes
# python $DROP_C @$STIMULI_ROOT/pilot-drop/pilot_it2_drop_sizes_box/commandline_args.txt --dir "D:\hsiaoyut\tdw_physics\data\drop\pilot_it2_drop_sizes_box" --num $NUM --only_use_flex_objects  --training_data_mode --seed 2 --save_passes "" --write_passes "_img,_id" --save_meshes


#python tdw_physics/target_controllers/drop.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-drop/iteration-1/pilot_it1_drop_all_bowls_box/commandline_args.txt --dir data/drop/pilot_it1_drop_all_bowls_box --save_meshes  --temp "D:/temp.hdf5"  --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/drop.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-drop/iteration-1/pilot_it1_drop_sidezone_box/commandline_args.txt --dir data/drop/pilot_it1_drop_sidezone_box  --save_meshes --temp "D:/temp.hdf5" --num 10 --height 128 --width 128
#python tdw_physics/target_controllers/drop.py @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-drop/pilot_it2_drop_simple_box/commandline_args.txt --dir data/drop/pilot_it1_drop_simple_box --save_meshes --temp "D:/temp.hdf5" --num 10 --height 128 --width 128



CLOTH_C=tdw_physics/target_controllers/cloth_sagging.py
python $CLOTH_C @/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-clothSagging/test10/commandline_args.txt --dir data/containment/pilot-clothSagging --save_meshes  --num 10 --height 128 --width 128 --port=1071

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
