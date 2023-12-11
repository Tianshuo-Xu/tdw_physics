#!/bin/bash

DATAROOT=${1:-'tdw_room_v3_stacking'}
ROOM=${2:-'box'} # [tdw_room, box, mm_craftroom_1b]
PORT=${3:-'1071'}
GPU=${4:-'0'}

#--validation_set --save_meshes
python tdw_physics/dataset_generation/scripts/generate_playroom_large.py --dir $DATAROOT --port $PORT --randomize_moving_object --split 0 --num_trials_per_scene 1000 --height 256 --width 256 --save_passes _img,_id,_flow --save_movies --gpu $GPU --num_views 4 --room $ROOM --remove_zone True --pscale 1.0 --tscale 1.0 --occlusion_scale 1.0 --num_distractors 0 --prot [0,180] --trot [0,180] --launch_build
