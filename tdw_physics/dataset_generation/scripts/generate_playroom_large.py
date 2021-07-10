import time
import os, sys
import random
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict
import copy
import json
from tdw_physics.util import MODEL_LIBRARIES
from tdw_physics.target_controllers.playroom import Playroom, get_playroom_args


RECORDS = []
for lib in MODEL_LIBRARIES.values():
    RECORDS.extend(lib.records)
MODELS = list(set([r for r in RECORDS if not r.do_not_use]))
MODEL_NAMES = sorted(list(set([r.name for r in RECORDS if not r.do_not_use])))
CATEGORIES = sorted(list(set([r.wcategory for r in RECORDS])))
MODELS_PER_CATEGORY = {k: [r for r in MODELS if r.wcategory == k] for k in CATEGORIES}
NUM_MODELS_PER_CATEGORY = {k: len([r for r in MODELS if r.wcategory == k]) for k in CATEGORIES}
NUM_MOVING_MODELS = 1000
NUM_STATIC_MODELS = 1000
SAVE_FRAME = 0

def get_args(dataset_dir: str):

    playroom_parser, playroom_postproc = get_playroom_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[playroom_parser], conflict_handler='resolve')
    
    parser.add_argument("--category_seed",
                        type=int,
                        default=0,
                        help="random seed for splitting categores")
    parser.add_argument("--models_per_split",
                        type=int,
                        default=200,
                        help="number of models in each category split")
    parser.add_argument("--num_moving_models",
                        type=int,
                        default=NUM_MOVING_MODELS,
                        help="number of models that will move")
    parser.add_argument("--num_static_models",
                        type=int,
                        default=NUM_STATIC_MODELS,
                        help="number of models that will be static")        

    args = parser.parse_args()
    args = playroom_postproc(args)
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

    rng = np.random.RandomState(seed)
    model_splits = OrderedDict()
    cat_split_ind = 0
    for i,num in enumerate(num_models_per_split):
        models_here = []        
        while len(models_here) < num:
            cats = category_splits[cat_split_ind]
            for cat in cats:
                models_here.extend(MODELS_PER_CATEGORY[cat])
            cat_split_ind += 1
        model_splits[i] = rng.permutation([r.name for r in models_here])[:num]

    return model_splits

def main(args):
    category_splits = make_category_splits(
        seed=args.category_seed,
        num_per_split=args.models_per_split,
    )
    model_splits = split_models(category_splits, [args.num_moving_models, args.num_static_models],
                                seed=args.category_seed)

    print([len(model_splits[i]) for i in [0,1]])

    

if __name__ == '__main__':

    args = get_args("playroom_large")
    main(args)

    



