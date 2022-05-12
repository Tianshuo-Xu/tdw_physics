#!/bin/bash
DATAROOT=${1:-'tdw_zoo_10obj'}
PORT=${2:-'1071'}
while true; do
  kill -9 `ps -aux | grep "TDW.x86_64" | awk '{print $2}'`
	python tdw_physics/dataset_generation/scripts/generate_playroom_large.py --dir $DATAROOT --port $PORT --randomize_moving_object --split 0 --height 256 --width 256 --save_passes _img,_id,_flow --save_movies --gpu 0 --num_views 4 --room archviz_house --remove_zone True --pscale 1.7 --tscale 1.7 --occlusion_scale 1.7 --num_distractors 0 --prot None --trot None --save_meshes --launch_build  --room_center [-12.0,1.0,-5.0] &&
	break; done
