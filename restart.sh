#!/bin/bash
DATAROOT=${1:-'tdw_room_allobj'}
ROOM=${2:-'tdw_room'}
PORT=${3:-'1071'}
while true; do
  kill -9 `ps -aux | grep "TDW.x86_64" | awk '{print $2}'`
	python tdw_physics/dataset_generation/scripts/generate_playroom_large.py --dir $DATAROOT --port $PORT --randomize_moving_object --split 0 --height 256 --width 256 --save_passes _img,_id,_flow --save_movies --gpu 3 --num_views 4 --room $ROOM --remove_zone True --pscale 1.0 --tscale 1.0 --occlusion_scale 1.0 --num_distractors 0 --prot None --trot None --launch_build  &&
	break; done
