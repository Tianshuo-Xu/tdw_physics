#! /usr/bin/env bash
#
# Run parallel commands and fail if any of them fails.
#
ROOM_1=${1:-'box'}
ROOM_2=${2:-'archviz_hourse'}
GPU_1=${3:-'3'}
GPU_2=${4:-'4'}
set -eu

pids=()

run_command () {
  kill -9 `ps -aux | grep TDW.x86_64 | awk '{print $2}'`
  kill -9 `ps -aux | grep generate_playroom | awk '{print $2}'`
  for x in 0 1 2 3; do
    if [[ $x -gt 3 ]]
    then
      ./generate.sh $ROOM_2 $(($x-4)) $((9271+$x)) $GPU_2 &
      pids+=($!)
    else
      ./generate.sh $ROOM_1 $x $((9371+$x)) $GPU_1 &
      pids+=($!)
    fi
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

while true; do
  run_command &&
	break;
done