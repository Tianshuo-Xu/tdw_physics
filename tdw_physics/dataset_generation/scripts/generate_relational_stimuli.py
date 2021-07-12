import time
import os, sys
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import json
import random

import tdw_physics.target_controllers.relations as rel
from tdw_physics.dataset_generation.configs.relational_experimental import (TRAIN_SCENARIOS, TRAIN_SINGLE,
                                                                            TEST_SCENARIOS, TEST_SINGLE)

RERUN = False
RERUN_SCENARIOS = json.loads(Path('.').joinpath('rerun.json').read_text())
SAVE_FRAME = -1

def get_scenario_pathname(container, target, distractor, relation_type):
    return "%s_from_%s_to_%s_with_%s" % (relation_type, container, target, distractor)

def main(output_dir, num, controller_args, launch_build=True, train=True, rerun=False):

    args = controller_args
    if rerun:
        scenarios = RERUN_SCENARIOS
    elif args.single_object:
        scenarios = [TEST_SINGLE, TRAIN_SINGLE][int(train)]
    else:
        scenarios = [TEST_SCENARIOS, TRAIN_SCENARIOS][int(train)]

    num_scenarios = len(scenarios)
    print("NUM SCENARIOS: %d" % num_scenarios)

    seed = args.seed - 1
    start = (args.start or 0)
    end = (args.end or num_scenarios)
    total = end - start
    temp_path = 'tmp' + str(args.gpu)
    seed = args.seed - 1 + (start * len(rel.Relation))

    for i, sc in enumerate(scenarios[start:end]):

        ((con, tar, dist), kwargs) = sc

        rels = kwargs.get('relations', [r.name for r in rel.Relation])
        rels = [r for r in rel.Relation if r.name in rels]

        if i == 0:
            rc = rel.RelationArrangement(
                launch_build=launch_build,
                port=args.port,
                randomize=0,
                seed=seed,

                ## positions
                single_object=args.single_object,
                no_object=args.no_object,
                container_position_range=args.cposition,
                target_position_range=args.tposition,
                target_rotation_range=args.trotation,
                target_angle_range=args.tangle,
                target_position_jitter=args.tjitter,
                target_always_horizontal=args.thorizontal,

                ## scales
                zone_scale_range=args.zscale,
                container_scale_range=args.cscale,
                target_scale_range=args.tscale,
                distractor_scale_range=args.dscale,
                max_target_scale_ratio=args.max_target_scale_ratio,

                ## camera
                camera_radius=args.camera_distance,
                camera_min_angle=args.camera_min_angle,
                camera_max_angle=args.camera_max_angle,
                camera_min_height=args.camera_min_height,
                camera_max_height=args.camera_max_height,
                camera_left_right_reflections=args.camera_left_right_reflections,

                ## common
                room=args.room,
                flex_only=False
            )

            rc._height, rc._width, rc._framerate = args.height, args.width, args.framerate
            if not Path(output_dir).exists():
                Path(output_dir).mkdir(parents=True)
            rc.command_log = Path(output_dir).joinpath('tdw_commands.json')
            rc.write_passes = ["_img", "_id", "_flow"]
            rc.save_passes = ["_img"]
            rc.save_movies = True
            rc.save_meshes = False
            rc.save_labels = False

            init_cmds = rc.get_initialization_commands(width=args.width, height=args.height)
            rc.communicate(init_cmds)

        for r in rels:

            print("Scenario %d of [%d, %d]" % (i + start + 1, start + 1, end))
            print("container: %s, target: %s, distractor: %s" % (con, tar, dist))
            print("relation type: %s" % r.name)
            output_path = os.path.join(output_dir,
                                       get_scenario_pathname(con, tar, dist, r.name))
            if not Path(output_path).exists():
                Path(output_path).mkdir(parents=True)
            rc.command_log = Path(output_path).joinpath("tdw_commands.json")

            rc.seed += 1
            rc.clear_static_data()
            rc.set_relation_types([r])
            rc.set_container_types([con])
            rc.set_target_types([tar])
            rc.set_distractor_types([dist])

            rc.trial_loop(
                num, output_dir=output_path, temp_path=temp_path, save_frame=SAVE_FRAME)


    rc.communicate({"$type": "terminate"})

if __name__ == '__main__':

    args = rel.get_relational_args("relational")

    import platform, os
    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":0." + str(args.gpu)
        else:
            os.environ["DISPLAY"] = ":0"

    main(args.dir, args.num, controller_args=args, train=bool(args.training_data), rerun=RERUN)
