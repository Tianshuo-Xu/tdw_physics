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
from tdw_physics.util import MODEL_LIBRARIES, ROOMS, none_or_str, none_or_int
from tdw_physics.target_controllers.playroom import Playroom, get_playroom_args
from tdw_physics.dataset_generation.scripts.playroom_selection import (
    EXCLUDE_MODEL, EXCLUDE_CATEGORY, CATEGORY_SCALE_FACTOR, MODEL_SCALE_FACTOR)
import h5py


RECORDS = []
for lib in {k:MODEL_LIBRARIES[k] for k in ["models_full.json", "models_special.json"]}.values():
    RECORDS.extend(lib.records)


MODELS = list(set([r for r in RECORDS if not r.do_not_use and not r.name in EXCLUDE_MODEL and not 'composite' in r.name]))
MODEL_NAMES = sorted(list(set([r.name for r in MODELS])))

CATEGORIES = sorted(list(set([r.wcategory for r in RECORDS])))
CATEGORIES = [cat for cat in CATEGORIES if cat not in EXCLUDE_CATEGORY]

MODELS_PER_CATEGORY = {k: [r for r in MODELS if r.wcategory == k] for k in CATEGORIES}
MODEL_NAME_PER_CATEGORY = {k: [r.name for r in MODELS if r.wcategory == k] for k in CATEGORIES}
NUM_MODELS_PER_CATEGORY = {k: len([r for r in MODELS if r.wcategory == k]) for k in CATEGORIES}

SCALE_DICT = {}
for category in CATEGORY_SCALE_FACTOR.keys():
    assert category in MODEL_NAME_PER_CATEGORY.keys(), category

for category, model_names in MODEL_NAME_PER_CATEGORY.items():
    if category in CATEGORY_SCALE_FACTOR.keys():
        factor = CATEGORY_SCALE_FACTOR[category]
    else:
        factor = 1.0

    for model in model_names:
        SCALE_DICT[model] = factor

for model, factor in MODEL_SCALE_FACTOR.items():
    SCALE_DICT[model] = factor


NUM_MOVING_MODELS = 1000
NUM_STATIC_MODELS = 1000
NUM_TOTAL_MODELS = len(MODEL_NAMES)
SAVE_FRAME = 0

# All the models with flex enabled
ALL_FLEX_MODELS = [r.name for r in RECORDS if (r.flex == True and not r.do_not_use and not r.name in EXCLUDE_MODEL and not 'composite' in r.name)]
for model in ALL_FLEX_MODELS:
    assert model in MODEL_NAMES, breakpoint()

rng = np.random.RandomState(seed=0)
HOLD_OUT_MODELS = [r for r in rng.choice(ALL_FLEX_MODELS, 200) if r in SCALE_DICT.keys()][0:100]
TRAIN_VAL_MODELS = [r for r in MODELS if r.name not in HOLD_OUT_MODELS and r.name in SCALE_DICT.keys()]

# TRAIN_VAL_CATEGORIES = [r.wcategory for r in TRAIN_VAL_MODELS]
# for r in RECORDS:
#     if r.name in HOLD_OUT_MODELS:
#         if r.wcategory in TRAIN_VAL_CATEGORIES:
#             print(r.name, r.wcategory)
#         else:
#             print('no', r.name, r.wcategory)
# breakpoint()
VAL_MODELS = [r.name for r in TRAIN_VAL_MODELS if r.name in ALL_FLEX_MODELS]
TRAIN_VAL_MODELS_NAMES = [r.name for r in TRAIN_VAL_MODELS]

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
    parser.add_argument("--load_cfg_path",
                        type=str,
                        default=None,
                        help="load_cfg_from_folder")
    parser.add_argument("--num_static_models",
                        type=int,
                        default=NUM_STATIC_MODELS,
                        help="number of models that will be static")
    parser.add_argument("--num_trials_per_model",
                        type=int,
                        default=10,
                        help="Number of trials to create per moving model")
    parser.add_argument("--num_trials_per_scene",
                        type=int,
                        default=10,
                        help="Number of trials to create per scene")
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
    parser.add_argument("--testing_set",
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

def build_simple_scenario(models, num_trials, seed, num_distractors, room, permute=True, load_cfg=None):
    room_seed = ROOMS.index(room)
    print('ROOM SEED', room_seed, seed)
    rng = np.random.RandomState(seed=(seed + room_seed))


    scenarios = []
    for i in range(num_trials):
        if load_cfg is not None:
            f = h5py.File(os.path.join(load_cfg, f'sc{i:04d}.hdf5'), "r")
            breakpoint()

        else:
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

def build_controller(args, scale_dict, launch_build=True):

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
        num_views=args.num_views,
        start=args.start,
        scale_factor_dict=scale_dict,
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

    # ## create the scenarios
    # if not args.validation_set:
    #     moving_models = moving_splits[args.split]
    #     static_models = static_splits[args.split % num_static_splits] if not args.use_all_static_models else all_static_models
    # else:
    #     print("remaining", remaining_splits)
    #     validation_models = []
    #     for models in remaining_splits:
    #         validation_models.extend(models)
    #     moving_models = static_models = validation_models

    # scenarios = build_scenarios(moving_models, static_models,
    #                             args.num_trials_per_model, seed=args.category_seed,
    #                             group_order=args.group_order,
    #                             randomize_moving_object=args.randomize_moving_object
    #                             )


    # models_simple = ['b06_train', 'emeco_su_bar', '699264_shoppingcart_2013', 'set_of_towels_roll']
    # models_simple = ['green_side_chair', 'red_side_chair', 'linen_dining_chair']
    # models_simple = ['cube'] * 4
    # models_simple = ['b03_zebra', 'checkers', 'cgaxis_models_50_24_vray']
    # 10obj zoo

    zoo_scale_dict = {
        'labrador_retriever_puppy': 1.,
        'b05_grizzly': 1.,
        'b04_horse_body_mesh': 1.,
        'b03_zebra_body': 1.,
        'b03_calf': 1.,
        '688926_elephant': 1.,
        '129601_sheep': 1.,
        'b03_dove_polysurface1': 1.,
        'b05_figure_2_node': 1.,
        'b04_duck': 1.
    }

    test_zoo_scale_dict = {
        '736684_elephant': 1.0,
        'b03_cow': 1.0,
        'b03_horse': 1.0,
        'rockdove_polysurface77': 1.0,
        'b04_mesh_giraffe': 1.0
    }

    kitchen_scale_dict = {
        'b04_orange_00': 0.3,
        'appliance-ge-profile-microwave3': 0.8,
        'b01_croissant': 0.8,
        'b03_banana_01_high': 1.0,
        'b03_cocacola_can_cage': 0.6,
        'b03_pcylinder2': 0.8,
        'b03_pink_donuts_mesh': 0.8,
        'coffee_maker': 0.7,
        'coffeemug': 0.6,
        'kettle': 0.7,
    }

    test_kitchen_scale_dict = {
        'b03_pain_au_chocolat': 0.8,
        'b04_chocolate_donuts_mesh': 0.8,
        'b03_rectangle01': 0.8,
        # 'apple': 0.3,
        'can_pepsi': 0.6
    }

    office_scale_dict = {
        'dice': 0.35,
        'arflex_strips_sofa': 1.0,
        'b03_worldglobe': 0.7,
        'b05_gym_matrix_t7xi_treadmill': 0.9,
        'b04_03_077': 0.6,
        'b04_vm_v2_025': 0.65,
        # 'b05_02_088': 0.8,
        'rucksack': 0.8,
        'b05_calculator': 0.7,
        'student_classical_guitar': 1.2,
        'buddah': 0.8
    }

    # if 'zoo_10obj_test' in args.dir:
    #     models_simple = list(test_zoo_scale_dict.keys())
    #     scale_dict = test_zoo_scale_dict
    # elif '20obj_test' in args.dir:
    #     models_simple =  list(test_zoo_scale_dict.keys()) +  list(test_kitchen_scale_dict.keys())
    #     scale_dict = test_zoo_scale_dict
    #     scale_dict.update(test_kitchen_scale_dict)
    # elif 'zoo_10obj' in args.dir:
    #     models_simple = list(zoo_scale_dict.keys())
    #     scale_dict = zoo_scale_dict
    # elif '20obj' in args.dir:
    #     models_simple =  list(zoo_scale_dict.keys()) +  list(kitchen_scale_dict.keys())
    #     scale_dict = zoo_scale_dict
    #     scale_dict.update(kitchen_scale_dict)
    # elif '30obj' in args.dir:
    #     models_simple = list(zoo_scale_dict.keys()) + list(kitchen_scale_dict.keys()) + list(office_scale_dict)
    #     scale_dict = zoo_scale_dict
    #     scale_dict.update(kitchen_scale_dict)
    #     scale_dict.update(office_scale_dict)
    # elif 'allobj' in args.dir:
    #     models_simple = TRAIN_VAL_MODELS_NAMES
    #     scale_dict = SCALE_DICT

    # else:
    #     raise ValueError


    scale_dict = SCALE_DICT

    # ['b05_02_088', '013_vray', 'giraffe_mesh', 'iphone_5_vr_white']
    # models_simple = ['b03_zebra', 'checkers', 'cgaxis_models_50_24_vray', 'b05_02_088', '013_vray', 'b03_852100_giraffe', 'iphone_5_vr_white', 'green_side_chair', 'red_side_chair', 'linen_dining_chair']
    # models_simple = static_models # ['green_side_chair', 'red_side_chair', 'linen_dining_chair']

    if args.testing_set:
        models_simple = HOLD_OUT_MODELS
        print('Generate test with %d models' % len(models_simple))
    elif args.validation_set:
        models_simple = VAL_MODELS
        print('Generate validation with %d models' % len(models_simple))
    else:
        models_simple = TRAIN_VAL_MODELS_NAMES
        print('Generate train with %d models' % len(models_simple))




    # for i,sc in enumerate(scenarios[start:end]):
    #     print(i, sc)

    ## set up the trial loop
    def _get_suffix(split, group_order):
        ns = num_moving_splits
        if group_order is None:
            group_order = range(4)
        suffix = split + ns * group_order[0]
        suffix = str(suffix % (ns * len(group_order)))
        return suffix

    suffix = _get_suffix(args.split, args.group_order) if not args.validation_set else 'val'
    output_dir = Path(args.dir) # .joinpath('model_split_' + suffix)
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

    exclude_list = ['archviz_house', 'box_room_4x5', 'tdw_room_4x5', 'suburb_scene_2018', 'savanna_flat_6km', 'mm_kitchen_4a', 'mm_kitchen_1b_4x5']


    rooms_list = []
    for i in ROOMS:
        if i not in exclude_list:
            rooms_list.append(i)

    if args.room == 'random':
        num_rooms = len(rooms_list)
    else:
        num_rooms = 1

    Play = build_controller(args, scale_dict)
    Play._height, Play._width, Play._framerate = (args.height, args.width, args.framerate)
    Play.command_log = output_dir.joinpath('tdw_commands.json')
    Play.write_passes = args.write_passes.split(',')
    Play.save_passes = args.save_passes.split(',')
    Play.save_movies = args.save_movies
    Play.save_meshes = args.save_meshes
    Play.save_labels = False
    count = 0
    for i in range(num_rooms):
        args.room = Play.room = rooms_list[i]

        output_dir = Path(os.path.join(args.dir, args.room))  # .joinpath('model_split_' + suffix)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        if 'floorplan_2' in args.room:
            Play.room_center_range = {'x':(7.2, 7.2), 'y': (0,0), 'z': (-2.5,-2.5)}
        elif 'floorplan_1' in args.room:
            Play.room_center_range = {'x': (0, 0), 'y': (0, 0), 'z': (2.5, 2.5)}
        elif 'floorplan_4' in args.room:
            Play.room_center_range = {'x': (-6.2, -6.2), 'y': (0, 0), 'z': (-3, -3)}
        elif 'floorplan_5' in args.room:
            Play.room_center_range = {'x': (-2.5, -2.5), 'y': (0, 0), 'z': (-2.5, -2.5)}
        elif 'house' in args.room:
            # [-12.8730,1.85,-5.75]
            Play.room_center_range = {'x': (12.8, 12.8), 'y': (1.85, 1.85), 'z': (-5.75, -5.75)}
        elif 'downtown' in args.room:
            Play.room_center_range = {'x': (4,-4), 'y': (-3., -3.), 'z': (-5, -5)}
        elif 'ruin' in args.room:
            Play.room_center_range = {'x': (0,0), 'y': (0,0), 'z': (-2.5, -2.5)}
        elif 'building_site' in args.room:
            Play.room_center_range = {'x': (0, 0), 'y': (0, 0), 'z': (2.5, 2.5)}
        elif 'craftroom_4' in args.room or 'kitchen_4' in args.room:
            Play.room_center_range = {'x': (0, 0), 'y': (0, 0), 'z': (0.5, 0.5)}
        elif 'craftroom_3' in args.room or 'kitchen_3' in args.room:
            Play.room_center_range = {'x': (0.25, 0.25), 'y': (0, 0), 'z': (0, 0)}
        else:
            Play.room_center_range = {'x': (0, 0), 'y': (0, 0), 'z': (0, 0)}
        start, end = args.start, (args.end or args.num_trials_per_scene)
        # Play.start = start
        # Play.end = end
        Play.communicate(Play.get_scene_initialization_commands())
        scenarios = build_simple_scenario(copy.deepcopy(models_simple), num_trials=args.num_trials_per_scene, seed=args.category_seed,
                                          num_distractors=args.num_distractors, room=args.room, permute=True,
                                          load_cfg=args.load_cfg_path)

        print('Number of models: ', len(models_simple))
        print('ROOM: ', args.room)

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
        count += 1


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

    print('USE GPU: ', args.gpu)

    main(args)

    # for nm in MODEL_NAMES:
    #     ok = _record_usable(nm)
    #     if not ok:
    #         print("%s has a bad non-composite sibling" % nm)
