#!/bin/bash

DATAROOT=${1:-'tdw_room_allobj'}
ROOM=${2:-'floorplan_1a'} # [tdw_room, box, mm_craftroom_1b]
PORT=${3:-'1071'}
GPU=${4:-'0'}

python tdw_physics/dataset_generation/scripts/generate_playroom_large.py \
--load_cfg_path '/ccn2/u/honglinc/datasets/tdw_playroom_v2_train/tdw_room_allobj' \
--dir $DATAROOT --port $PORT --randomize_moving_object --split 0 --height 256 --width 256 --save_passes _img,_id,_flow --save_movies --gpu $GPU --num_views 4 --room $ROOM --remove_zone True --pscale 1.0 --tscale 1.0 --occlusion_scale 1.0 --num_distractors 0 --prot None --trot None --launch_build
