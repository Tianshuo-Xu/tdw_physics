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
from tdw_physics.util import MODEL_LIBRARIES, none_or_str, none_or_int
from tdw_physics.target_controllers.playroom import Playroom, get_playroom_args


RECORDS = []
for lib in {k:MODEL_LIBRARIES[k] for k in ["models_full.json", "models_special.json"]}.values():
    RECORDS.extend(lib.records)
MODELS = list(set([r for r in RECORDS if not r.do_not_use]))
MODEL_NAMES = sorted(list(set([r.name for r in RECORDS if not r.do_not_use])))
CATEGORIES = sorted(list(set([r.wcategory for r in RECORDS])))
MODELS_PER_CATEGORY = {k: [r for r in MODELS if r.wcategory == k] for k in CATEGORIES}
NUM_MODELS_PER_CATEGORY = {k: len([r for r in MODELS if r.wcategory == k]) for k in CATEGORIES}
NUM_MOVING_MODELS = 1000
NUM_STATIC_MODELS = 1000
NUM_TOTAL_MODELS = len(MODEL_NAMES)
SAVE_FRAME = 0

def _record_usable(record_name):
    if 'composite' in record_name:
        non_composite_name = record_name.split('_composite')[0]
        non_composite_records = [r for r in RECORDS if r.name == non_composite_name]
        not_usable = any([r.do_not_use for r in non_composite_records])
        if not_usable:
            return False
    return True

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

    parser.add_argument("--category_seed",
                        type=int,
                        default=0,
                        help="random seed for splitting categores")
    parser.add_argument("--models_per_split",
                        type=int,
                        default=125,
                        help="number of models in each category split")
    parser.add_argument("--num_moving_models",
                        type=int,
                        default=NUM_MOVING_MODELS,
                        help="number of models that will move")
    parser.add_argument("--num_static_models",
                        type=int,
                        default=NUM_STATIC_MODELS,
                        help="number of models that will be static")
    parser.add_argument("--num_trials_per_model",
                        type=int,
                        default=10,
                        help="Number of trials to create per moving model")
    parser.add_argument("--split",
                        type=int,
                        default=0,
                        help="Which split of the trials to generate")
    parser.add_argument("--use_all_static_models",
                        action="store_true",
                        help="Whether to lump all the static models together for every split")
    parser.add_argument("--validation_set",
                        action="store_true",
                        help="Create the validation set using the remaining models")
    parser.add_argument("--randomize_moving_object",
                        action="store_true",
                        help="Randomize which object gets pushed")
    parser.add_argument("--group_order",
                        type=str,
                        default="[0,1,2,3]",
                        help="Permutation of [probes, targets, distractors, occluders] to use in scenarios")
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
    parser.add_argument("--num_views",
                        type=int,
                        default=1,
                        help="Number of views / cameras")
    parser.add_argument("--num_distractors",
                        type=int,
                        default=1,
                        help="Number of distractors")

    args = parser.parse_args()
    args = playroom_postproc(args)
    if args.group_order is not None:
        args.group_order = json.loads(args.group_order)
    else:
        args.group_order = range(4)
    return args

def make_category_splits(categories=CATEGORIES, num_per_split=200, seed=0):
    rng = np.random.RandomState(seed)
    total_num_models = sum((NUM_MODELS_PER_CATEGORY[k] for k in categories))
    category_splits = OrderedDict()
    categories_left = copy.deepcopy(categories)
    models_left = total_num_models
    split_ind = 0
    while models_left > num_per_split:
        num_now = 0
        split_now = []
        while (num_now < num_per_split) or (models_left < (num_per_split - num_now)):
            ## choose a category
            cat = categories_left.pop(rng.choice(range(len(categories_left))))
            split_now.append(cat)
            num_now += NUM_MODELS_PER_CATEGORY[cat]
            models_left -= NUM_MODELS_PER_CATEGORY[cat]

        category_splits[split_ind] = split_now
        print("split num = %d; categories = %s; num_models = %d" % (split_ind, split_now, num_now))
        split_ind += 1

    category_splits[split_ind] = categories_left
    print("split num = %d; categories = %s; num_models = %d" % (split_ind, categories_left, sum(NUM_MODELS_PER_CATEGORY[cat] for cat in categories_left)))
    print("MODELS USED: %d" % sum((sum(NUM_MODELS_PER_CATEGORY[cat] for cat in split) for split in category_splits.values())))

    return category_splits

def split_models(category_splits, num_models_per_split=[1000,1000], seed=0):

    rng = np.random.RandomState(seed=seed)
    model_splits = OrderedDict()
    cat_split_ind = 0
    for i,num in enumerate(num_models_per_split):
        models_here = []
        while (len(models_here) < num) and (cat_split_ind < len(category_splits.keys())):
            cats = category_splits[cat_split_ind]
            for cat in sorted(cats):
                models_here.extend(MODELS_PER_CATEGORY[cat])
            cat_split_ind += 1

        model_splits[i] = rng.permutation(sorted([r.name for r in models_here]))[:num]

    return model_splits

def build_simple_scenario(models, num_trials, seed, num_distractors, permute=True):
    rng = np.random.RandomState(seed=seed)

    scenarios = []
    for i in range(num_trials):
        permute_models = rng.permutation(models) if permute else models

        scene = {
            'probe': permute_models[0],
            'target': permute_models[1],
            'occluder': permute_models[2]
        }

        if num_distractors > 0:
            scene['distractor'] = permute_models[3]

        scene['apply_force_to'] = 'probe'

        scenarios.append(scene)

    return scenarios


def build_scenarios(moving_models,
                    static_models,
                    num_trials_per_model,
                    seed=0,
                    group_order=None,
                    randomize_moving_object=False
                    ):

    if group_order is None:
        group_order = range(4)

    rng = np.random.RandomState(seed=(seed + group_order[0]))
    num = num_trials_per_model
    NM = len(moving_models)
    NS = len(static_models)

    probes = rng.permutation(moving_models)
    targets = rng.permutation(moving_models)
    distractors = rng.permutation(static_models)
    occluders = rng.permutation(static_models)
    groups = [probes, targets, distractors, occluders]

    print("group order", group_order)
    probes, targets, distractors, occluders = [groups[g] for g in group_order]

    # ok_objects = {
    #     'probe': [nm for nm in probes if _record_usable(nm)],
    #     'target': [nm for nm in targets if _record_usable(nm)],
    #     'distractor': [nm for nm in distractors if _record_usable(nm)],
    #     'occluder': [nm for nm in occluders if _record_usable(nm)]
    # }

    scenarios = []
    for i in range(num * NM):
        probe_ind = i // num
        target_ind = i % NM
        dist_ind = occ_ind = i % NS
        scene = {
            'probe': probes[probe_ind],
            'target': targets[target_ind],
            'distractor': distractors[dist_ind],
            'occluder': occluders[occ_ind]
        }
        # for k in scene:
        #     if not _record_usable(scene[k]):
        #         nm = scene[k] + ''
        #         scene[k] = ok_objects[k][i*7 % len(ok_objects[k])]
        #         print("Substituted <<%s>> for <<%s>> as the %s object in trial %d" % \
        #                      (scene[k], nm, k, i))

        if randomize_moving_object:
            scene['apply_force_to'] = ['probe', 'target', 'distractor', 'occluder'][i % 4]
        else:
            scene['apply_force_to'] = 'probe'

        scenarios.append(scene)

    print("num, NM, NS", num, NM, NS)


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
        zone_material=args.zmaterial,
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
        target_color=args.color,
        probe_color=args.pcolor,
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
        target_material=args.tmaterial,
        probe_material=args.pmaterial,
        distractor_material=args.tmaterial,
        occluder_material=args.tmaterial,
        distractor_categories=None,
        num_distractors=args.num_distractors,
        occluder_categories=None,
        num_occluders=1,
        occlusion_scale=args.occlusion_scale,
        occluder_aspect_ratio=None,
        distractor_aspect_ratio=None,
        probe_lift = args.plift,
        flex_only=False,
        no_moving_distractors=args.no_moving_distractors,
        match_probe_and_target_color=args.match_probe_and_target_color,
        size_min=None,
        size_max=None,
        probe_initial_height=0.25,
        num_views=args.num_views
    )

    return C

def main(args):

    ## build the model splits
    category_splits = make_category_splits(
        seed=args.category_seed,
        num_per_split=args.models_per_split,
    )

    mps = args.models_per_split
    num_moving_splits = args.num_moving_models // mps
    num_static_splits = args.num_static_models // mps
    nums_per_split = [mps] * (num_moving_splits + num_static_splits)
    nums_per_split += [NUM_TOTAL_MODELS]
    model_splits = split_models(category_splits, nums_per_split,
                                seed=args.category_seed)
    moving_splits = [model_splits[i] for i in range(num_moving_splits)]
    static_splits = [model_splits[j] for j in range(num_moving_splits, num_moving_splits + num_static_splits)]
    remaining_splits = [model_splits[k] for k in range(num_moving_splits + num_static_splits, len(model_splits.keys()))]
    all_static_models = []
    for s in static_splits:
        all_static_models.extend(s)

    ## create the scenarios
    if not args.validation_set:
        moving_models = moving_splits[args.split]
        static_models = static_splits[args.split % num_static_splits] if not args.use_all_static_models else all_static_models
    else:
        print("remaining", remaining_splits)
        validation_models = []
        for models in remaining_splits:
            validation_models.extend(models)
        moving_models = static_models = validation_models

    # scenarios = build_scenarios(moving_models, static_models,
    #                             args.num_trials_per_model, seed=args.category_seed,
    #                             group_order=args.group_order,
    #                             randomize_moving_object=args.randomize_moving_object
    #                             )


    # models_simple = ['b06_train', 'emeco_su_bar', '699264_shoppingcart_2013', 'set_of_towels_roll']
    # models_simple = ['green_side_chair', 'red_side_chair', 'linen_dining_chair']
    # models_simple = ['cube'] * 4
    models_simple = ['b03_zebra', 'checkers', 'cgaxis_models_50_24_vray']
    # ['b05_02_088', '013_vray', 'giraffe_mesh', 'iphone_5_vr_white']
    # models_simple = ['b03_zebra', 'checkers', 'cgaxis_models_50_24_vray', 'b05_02_088', '013_vray', 'b03_852100_giraffe', 'iphone_5_vr_white', 'green_side_chair', 'red_side_chair', 'linen_dining_chair']
    # models_simple = static_models # ['green_side_chair', 'red_side_chair', 'linen_dining_chair']
    scenarios = build_simple_scenario(models_simple, num_trials=1000, seed=args.category_seed, num_distractors=args.num_distractors, permute=True)

    start, end = args.start, (args.end or len(scenarios))

    for i,sc in enumerate(scenarios[start:end]):
        print(i, sc)

    ## set up the trial loop
    def _get_suffix(split, group_order):
        ns = num_moving_splits
        if group_order is None:
            group_order = range(4)
        suffix = split + ns * group_order[0]
        suffix = str(suffix % (ns * len(group_order)))
        return suffix

    suffix = _get_suffix(args.split, args.group_order) if not args.validation_set else 'val'
    output_dir = Path(args.dir).joinpath('model_split_' + suffix)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    temp_path = Path('tmp' + str(args.gpu))
    if not temp_path.parent.exists():
        temp_path.parent.mkdir(parents=True)
    if temp_path.exists():
        temp_path.unlink()

    # log
    setup_logging(output_dir)
    logging.info("Generating split %s / %d with category seed %d" % \
                 (suffix, 2*(num_moving_splits + num_static_splits)-1, args.category_seed))
    logging.info("Using group order %s" % [['probes', 'targets', 'distractors', 'occluders'][g] for g in args.group_order])

    ## init the controller
    if (args.seed == -1) or (args.seed is None):
        args.seed = int(args.split) + num_moving_splits * args.group_order[0]


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
    Play.trial_loop(num=(end - start),
                    output_dir=str(output_dir),
                    temp_path=str(temp_path),
                    save_frame=SAVE_FRAME,
                    update_kwargs=scenarios[start:end],
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

    print("PUSHING WITH FORCE %.2f" % args.fscale[0])
    main(args)

    # for nm in MODEL_NAMES:
    #     ok = _record_usable(nm)
    #     if not ok:
    #         print("%s has a bad non-composite sibling" % nm)
