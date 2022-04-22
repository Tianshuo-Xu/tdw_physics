import time
import logging
import os, sys
import random
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict
import copy
import json
from tqdm import tqdm
from tdw.librarian import MaterialLibrarian, SceneLibrarian
from tdw_physics.util import MODEL_LIBRARIES, none_or_str, none_or_int
from tdw_physics.target_controllers.playroom import Playroom, get_playroom_args

EXCLUDE = ['platonic', 'dumbbell', 'pentagon']
RECORDS = []
for lib in {k:MODEL_LIBRARIES[k] for k in ["models_flex.json"]}.values():
    RECORDS.extend([r for r in lib.records if r.name not in EXCLUDE])
MODELS = list(set([r for r in RECORDS if not r.do_not_use]))
MODEL_NAMES = sorted(list(set([r.name for r in RECORDS if not r.do_not_use])))

M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES_BY_TYPE = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                          for mtype in MATERIAL_TYPES}
MATERIAL_NAMES = []
for mtype in sorted(MATERIAL_TYPES):
    MATERIAL_NAMES.extend(MATERIAL_NAMES_BY_TYPE[mtype])
    
SAVE_FRAME = 0

def setup_logging(logdir):

    logdir = Path(logdir)
    if not logdir.exists():
        logdir.parent.mkdir(parents=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s --- %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(str(logdir), "generation.log")),
                            logging.StreamHandler()
                        ])

def get_args(dataset_dir: str):

    playroom_parser, playroom_postproc = get_playroom_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[playroom_parser], conflict_handler='resolve')

    parser.add_argument("--material_seed",
                        type=int,
                        default=0,
                        help="random seed for splitting categores")
    parser.add_argument("--material_types",
                        type=none_or_str,
                        default=None,
                        help="Which class of materials to sample material names from")
    parser.add_argument("--probe",
                        type=none_or_str,
                        default=','.join(MODEL_NAMES),
                        help="Which class of materials to sample material names from")    
    parser.add_argument("--materials_per_split",
                        type=int,
                        default=60,
                        help="number of models in each materials split")
    parser.add_argument("--num",
                        type=int,
                        default=2,
                        help="Number of trials to create per material per model")
    parser.add_argument("--split",
                        type=int,
                        default=0,
                        help="Which split of the trials to generate")
    parser.add_argument("--validation_set",
                        action="store_true",
                        help="Create the validation set using the remaining models")
    parser.add_argument("--start",
                        type=int,
                        default=0,
                        help="Which scenario to start with")
    parser.add_argument("--end",
                        type=int,
                        default=None,
                        help="Which scenario to end with")
    parser.add_argument("--launch_build",
                        action="store_true",
                        help="Whether to launch the build")

    ## size ranges for objects    
    parser.add_argument("--pscale",
                        type=str,
                        default="[[0.1,1.0],[0.1,1.0],[0.1,1.0]]",
                        help="scale of probe objects")
    parser.add_argument("--tscale",
                        type=str,
                        default="[[0.1,1.0],[0.1,1.0],[0.1,1.0]]",                        
                        help="scale of target objects")    
    parser.add_argument("--size_min",
                        type=none_or_str,
                        default="0.05",
                        help="Minimum size for probe and target objects")
    parser.add_argument("--size_max",
                        type=none_or_str,
                        default="1.25",
                        help="Maximum size for probe and target objects")    

    args = parser.parse_args()
    args = playroom_postproc(args)

    return args

def make_material_splits(material_types=MATERIAL_TYPES, num_per_split=50, seed=0):
    rng = np.random.RandomState(seed)
    materials = []
    for mtype in sorted(material_types):
        materials.extend(MATERIAL_NAMES_BY_TYPE[mtype])
    materials = rng.permutation(materials)
    
    num_splits = int(np.ceil(len(materials) / float(num_per_split)))
    material_splits = OrderedDict()
    materials_left = len(materials)

    split_ind = 0
    while materials_left > num_per_split:
        ms, me = [split_ind * num_per_split, (split_ind + 1) * num_per_split]
        material_splits[split_ind] = materials[ms:me]
        materials_left -= len(material_splits[split_ind])        
        split_ind += 1
        print("split num = %d; materials = %s; num_materials = %d" % \
              (split_ind - 1, material_splits[split_ind-1], len(material_splits[split_ind-1])))

    material_splits['validation'] = materials[me:]

    return material_splits

def build_scenarios(materials, models, num_trials_per_material_per_model, seed=0):
    """
    Create num_trials_per_material_per_model ... per material per model.
    """
    rng = np.random.RandomState(seed=seed)
    num_materials = len(materials)
    num_models = len(models)
    num_per = num_trials_per_material_per_model    
    num_trials = num_materials * num_models * num_trials_per_material_per_model

    def concat_permuted_lists(my_list, num):
        lists = [my_list for _ in range(num)]
        cat = []
        for x in lists:
            cat.extend(rng.permutation(x))
        return cat

    probe_materials = rng.permutation(materials)
    probe_models = rng.permutation(models)

    scenarios = []
    for m in range(num_materials):
        target_materials = concat_permuted_lists(materials, num=(num_models * num_per)) # len -> num_trials
        distractor_materials = concat_permuted_lists(materials, num=(num_models * num_per))
        occluder_materials = concat_permuted_lists(materials, num=(num_models * num_per))
        zone_materials = concat_permuted_lists(materials, num=(num_models * num_per))
        for o in range(num_models):
            target_models = concat_permuted_lists(models, num=num_per) # len -> num_models * num_per
            distractor_models = concat_permuted_lists(models, num=num_per)
            occluder_models = concat_permuted_lists(models, num=num_per)            
            for p in range(num_per):

                model_ind = p + o*num_per
                material_ind = p + o*num_per + m*num_per*num_models

                ## populate object models
                scene = {
                    'probe': probe_models[o],
                    'target': target_models[model_ind],
                    'distractor': distractor_models[model_ind],
                    'occluder': occluder_models[model_ind]
                }

                ## populate materials
                scene.update({
                    'probe_material': probe_materials[m],
                    'target_material': target_materials[material_ind],
                    'distractor_material': distractor_materials[material_ind],
                    'occluder_material': occluder_materials[material_ind],
                    'zone_material': zone_materials[material_ind]
                })

                scenarios.append(scene)

    for i,scene in enumerate(scenarios):
        print("scene %d: probe_material = %s, target_material = %s, distractor_material = %s, occluder_material = %s" % (i, scene['probe_material'], scene['target_material'], scene['distractor_material'], scene['occluder_material']))
        print("scene %d: probe = %s, target = %s, distractor = %s, occluder = %s" %
              (i, scene['probe'], scene['target'], scene['distractor'], scene['occluder']))                

    return scenarios


def build_controller(args, launch_build=True):

    C = Playroom(
        launch_build=args.launch_build,
        port=args.port,
        room=args.room,
        room_center_range=args.room_center,
        randomize=0,
        seed=args.seed,
        target_zone=args.zone,
        zone_location=args.zlocation,
        zone_scale_range=args.zscale,
        zone_color=None,
        zone_material=None,
        zone_friction=args.zfriction,
        target_objects=args.target,
        target_categories=args.target_categories,
        probe_objects=args.probe,
        probe_categories=args.probe_categories,
        target_scale_range=args.tscale,
        target_rotation_range=args.trot,
        probe_rotation_range=args.prot,
        probe_scale_range=args.pscale,
        probe_mass_range=args.pmass,
        target_color=None,
        probe_color=None,
        collision_axis_length=args.collision_axis_length,
        force_scale_range=args.fscale,
        force_angle_range=args.frot,
        force_offset=args.foffset,
        force_offset_jitter=args.fjitter,
        force_wait=args.fwait,
        remove_target=bool(args.remove_target),
        remove_zone=bool(args.remove_zone),
        zjitter = args.zjitter,
        fupforce = args.fupforce,
        ## not scenario-specific
        camera_radius=args.camera_distance,
        camera_min_angle=args.camera_min_angle,
        camera_max_angle=args.camera_max_angle,
        camera_min_height=args.camera_min_height,
        camera_max_height=args.camera_max_height,
        monochrome=args.monochrome,
        material_types=args.material_types,
        target_material=None,
        probe_material=None,
        distractor_categories=None,
        num_distractors=1,
        occluder_categories=None,
        num_occluders=1,
        occlusion_scale=args.occlusion_scale,
        occluder_aspect_ratio=None,
        distractor_aspect_ratio=None,
        probe_lift = args.plift,
        flex_only=False,
        no_moving_distractors=args.no_moving_distractors,
        match_probe_and_target_color=False,
        size_min=args.size_min,
        size_max=args.size_max,
        probe_initial_height=0.25,
        randomize_object_size=True,
        occluder_material=None,
        distractor_material=None,
        model_libraries=["models_flex.json"]
    )

    return C

def main(args):

    ## build the material splits
    material_splits = make_material_splits(
        material_types=args.material_types,
        num_per_split=args.materials_per_split,
        seed=args.material_seed)


    if not args.validation_set:
        materials = material_splits[args.split]
    else:
        materials = material_splits['validation']
    
    scenarios = build_scenarios(materials,
                                models=args.probe,
                                num_trials_per_material_per_model=args.num,
                                seed=args.material_seed)

    ## set up the trial loop
    suffix = str(args.split) if not args.validation_set else 'val'
    output_dir = Path(args.dir).joinpath('material_split_' + suffix)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    temp_path = Path('tmp' + str(args.gpu))
    if not temp_path.parent.exists():
        temp_path.parent.mkdir(parents=True)
    if temp_path.exists():
        temp_path.unlink()

    ## log
    setup_logging(output_dir)
    logging.info("Generating split %d / %d with materials seed %d" % \
                 (args.split, len(material_splits.keys()) - 1, args.material_seed))

    ## init the controller
    if (args.seed == -1) or (args.seed is None):
        args.seed = int(args.split)
    Play = build_controller(args)
    Play._height, Play._width, Play._framerate = (args.height, args.width, args.framerate)
    Play.command_log = output_dir.joinpath('tdw_commands.json')
    Play.write_passes = args.write_passes.split(',')
    Play.save_passes = args.save_passes.split(',')
    Play.save_movies = args.save_movies
    Play.save_meshes = False
    Play.save_labels = False

    log_cmds = [{"$type": "set_network_logging", "value": True}]
    init_cmds = Play.get_initialization_commands(width=args.width, height=args.height)
    Play.communicate(log_cmds + init_cmds)
    logging.info("Initialized Controller with random seed %d" % args.seed)

    ## run the trial loop
    Play.trial_loop(num=len(scenarios),
                    output_dir=str(output_dir),
                    temp_path=str(temp_path),
                    save_frame=SAVE_FRAME,
                    update_kwargs=scenarios,
                    unload_assets_every=args.unload_assets_every,
                    do_log=True)

    ## terminate build
    Play.communicate({"$type": "terminate"})

if __name__ == '__main__':

    args = get_args("playroom_large")

    import platform, os
    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":0." + str(args.gpu)
        else:
            os.environ["DISPLAY"] = ":0"

    main(args)
