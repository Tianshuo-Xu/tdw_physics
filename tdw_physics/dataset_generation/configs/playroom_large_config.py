import random
import numpy as np
from tdw_physics.util import MODEL_LIBRARIES

SEED = 0
RECORDS = []
for lib in MODEL_LIBRARIES.values():
    RECORDS.extend(lib.records)

MODELS = sorted(list(set([r.name for r in RECORDS if not r.do_not_use])))
CATEGORIES = sorted(list(set([r.wcategory for r in RECORDS])))
NUM_MOVING_MODELS = 1000
NUM_STATIC_MODELS = 1000



