#!/bin/bash

DATAROOT=${1:-'tdw_room_allobj'}
ROOM=${2:-'tdw_room'} # [tdw_room, box, mm_craftroom_1b]
PORT=${3:-'1071'}
MOVEMENT_SEED=${5:-'0'}
GPU=${4:-'0'}

#python tdw_physics/dataset_generation/scripts/generate_playroom_large.py --dir $DATAROOT --port $PORT --randomize_moving_object --split 0 --height 256 --width 256 --save_passes _img,_id,_flow --save_movies --gpu $GPU --num_views 4 --room $ROOM --remove_zone True --pscale 1.0 --tscale 1.0 --occlusion_scale 1.0 --num_distractors 0 --prot None --trot None --launch_build

python3 tdw_physics/dataset_generation/scripts/generate_playroom_large.py --dir $DATAROOT --port $PORT --randomize_moving_object --split 0 --height 128 --width 128 --save_passes _img,_id,_flow --save_movies --num_views 4 --room $ROOM --remove_zone True --pscale 1.0 --tscale 1.0 --occlusion_scale 1.0 --num_distractors 0 --prot None --trot None --launch_build --movement_seed $MOVEMENT_SEED