TRAIN_CONTAINERS = [
    "woven_box",
    "round_bowl_large_thin",
    "medium_mesh_basket",
    "box_tapered_beech",
    "basket_18inx18inx12iin_wood_mesh"
]

TRAIN_OBJECTS = [
    "b04_clownfish",
    "886673_duck",
    "b04_red_grapes",
    "b05_calculator",
    "half_circle_wood_block"    
]

TRAIN_SCENARIOS = []
for d in (TRAIN_OBJECTS + TRAIN_CONTAINERS):
    for c in TRAIN_CONTAINERS:
        for o in TRAIN_OBJECTS:
            sc = ((c, o, d), {})
            TRAIN_SCENARIOS.append(sc)

TRAIN_SINGLE = []
for c in TRAIN_CONTAINERS:
    for o in (TRAIN_OBJECTS + TRAIN_CONTAINERS):
        sc = ((c, o, o), {})
        TRAIN_SINGLE.append(sc)

TEST_CONTAINERS = [
    "b04_wicker_tray",
    "wooden_box",
    "small_mesh_basket",
    "b04_bowl_smooth",
    "measuring_pan"    
]

TEST_OBJECTS = [
    "b05_lobster",
    "b06_cat_1",
    "b05_banana_rig_2",
    "b05_workglove_rc01_zb05",
    "l-shape_wood_block"
]

TEST_SCENARIOS = []
for d in (TEST_OBJECTS + TEST_CONTAINERS):
    for c in TEST_CONTAINERS:
        for o in TEST_OBJECTS:
            sc = ((c, o, d), {})
            TEST_SCENARIOS.append(sc)

TEST_SINGLE = []
for c in TEST_CONTAINERS:
    for o in (TEST_OBJECTS + TEST_CONTAINERS):
        sc = ((c, o, o), {})
        TEST_SINGLE.append(sc)
