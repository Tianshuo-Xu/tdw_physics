#!/bin/bash

ROOM=${1:-'box'}
SPLIT=${2:-'0'}
PORT=${3:-'1071'}
GPU=${4:-'0'}
DATAROOT=${5:-'playroom_mv_v1'}

if [ "$ROOM" = "archviz_house" ]; then
    ROOM_CENTER=[-11.,0.96,-4.75]
else
    ROOM_CENTER=[0.,0.,0.]
fi

#while true; do
  python -u tdw_physics/dataset_generation/scripts/generate_playroom_large.py \
  --dir /data3/honglinc/tdw_datasets/$DATAROOT \
  --port $PORT \
  --randomize_moving_object \
  --split $SPLIT \
  --height 1024 --width 1024 \
  --save_passes _img,_id,_flow \
  --save_movies \
  --gpu $GPU \
  --num_views 4 \
  --room $ROOM \
  --remove_zone False \
  --pscale 1.0 --tscale 1.0 --occlusion_scale 1.0 \
  --num_distractors 0 --prot None --trot None \
  --launch_build \
  --validation_set \
  --num_trials_per_model 1 \
  --room_center $ROOM_CENTER #&&
#	break;
#done
