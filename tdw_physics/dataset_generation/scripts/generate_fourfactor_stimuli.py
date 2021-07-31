import time
import os, sys
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import json
import random
import copy
from collections import OrderedDict
import stopit

import tdw_physics.target_controllers.relations as rel
from tdw_physics.dataset_generation.configs.fourfactor_experimental import \
    (set_a_params, set_b_params,
     contain_params, occlude_params, collide_params, miss_params,
     tdwroom_params, suburb1_params, suburb2_params,
     common_params)

SAVE_FRAME = 90

def build_scenario_name(room, relation, collide, probe):
    name = [probe,
            "collide" if collide else "miss",
            relation,
            room + "room"]
    name = "_".join(name)
    return name

def build_scenarios():

    scenarios = OrderedDict()
    
    for r, rooms in enumerate([tdwroom_params, suburb1_params, suburb2_params]):
        params = dict()
        params.update(copy.deepcopy(common_params))
        room_params = copy.deepcopy(params)
        room_params.update(copy.deepcopy(rooms))
        for objs in [set_a_params, set_b_params]:
            obj_params = copy.deepcopy(room_params)
            obj_params.update(copy.deepcopy(objs))
            for i, con in enumerate([collide_params, occlude_params]):
                con_params = copy.deepcopy(obj_params)
                con_params.update(copy.deepcopy(con))
                con_params['relation'] = [rel.Relation.contain, rel.Relation.occlude][i]
                for j, coll in enumerate([collide_params, miss_params]):
                    coll_params = copy.deepcopy(con_params)
                    coll_params.update(copy.deepcopy(coll))
                    for k,v in coll_params.items():
                        if 'range' in k:
                            coll_params[k] = rel.handle_random_transform_args(v)
                            
                    name = build_scenario_name(
                        room=coll_params['room'].split('_')[0] + str(r),
                        relation=coll_params['relation'].name,
                        collide=(not bool(j)),
                        probe=coll_params['distractor']
                    )
                    scenarios[name] = copy.deepcopy(coll_params)
                    
    return scenarios

def build_controller(params):
    
    c = rel.RelationArrangement(
        launch_build=True,
        port=None,
        randomize=0,
        single_object=False,
        no_object=False,
        zone_scale_range=rel.handle_random_transform_args("-1.0"),
        zone_location={'x':10.0, 'y': 10.0, 'z': 10.0},
        flex_only=False,
        **params)

    c.write_passes = ["_img", "_id", "_flow"]
    c.save_passes = ["_img"]
    c.save_movies = True
    c.save_meshes = False
    c.save_labels = False

    return c

def main(args):

    scenarios = build_scenarios()
    for i,s in enumerate(scenarios.values()):
        print(i, list(scenarios.keys())[i])
    num_scenarios = len(scenarios)

    # seeding and indexing into scenarios
    start = (args.start or 0)
    end = (args.end or num_scenarios)
    total = end - start
    seed = args.seed + start
    temp_path = 'tmp' + str(args.gpu)

    for i, nm in enumerate(list(scenarios.keys())[start:end]):

        print("scenario %d: %s" % (i + start, nm))
        seed += 1        
        sc_params = scenarios[nm]
        sc_params['seed'] = seed
        rc = build_controller(sc_params)
        rc._height, rc._width, rc._framerate = args.height, args.width, args.framerate
        rc.clear_static_data()

        ## set output path and init
        output_dir = Path(args.dir).joinpath(nm)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        rc.command_log = output_dir.joinpath("tdw_commands.json")
        init_cmds = rc.get_initialization_commands(width=args.width, height=args.height)
        rc.communicate(init_cmds)

        rc.trial_loop(
            args.num, output_dir=str(output_dir), temp_path=temp_path, save_frame=SAVE_FRAME)

        with stopit.SignalTimeout(5) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
            end = rc.communicate({"$type": "terminate"})
            if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
                print("tdw closed successfully")
            elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                end = rc.communicate({"$type": "terminate"})                
                print("tdw failed to acknowledge being closed. tdw window might need to be manually closed")            
    
    return 0

if __name__ == '__main__':

    args = rel.get_relational_args("relational")

    import platform, os
    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":0." + str(args.gpu)
        else:
            os.environ["DISPLAY"] = ":0"

    main(args)
