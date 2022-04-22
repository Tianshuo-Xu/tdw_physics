import os, sys
import json
from pathlib import Path

if __name__ == '__main__':

    data1 = sys.argv[1]
    data2 = sys.argv[2]

    meta1 = os.path.join(data1, 'model_names_by_field.json')
    meta2 = os.path.join(data2, 'model_names_by_field.json')

    m1 = json.loads(Path(meta1).read_text(), encoding='utf-8')
    m2 = json.loads(Path(meta2).read_text(), encoding='utf-8')

    probes1 = set(m1['probe_name'])
    probes2 = set(m2['probe_name'])

    intersection = probes1.intersection(probes2)
    print("Number of moving models in set 1: %d" % len(probes1))
    print("Number of moving models in set 2: %d" % len(probes2))
    print("Number of moving models overlapping: %d" % len(intersection))

    all_models1 = m1['probe_name']
    for m in m1['static_model_names']:
        all_models1.extend(m)
    all_models1 = set(all_models1)

    all_models2 = m2['probe_name']
    for m in m2['static_model_names']:
        all_models2.extend(m)
    all_models2 = set(all_models2)
    intersection = all_models1.intersection(all_models2)
    print("Number of models in set 1: %d" % len(all_models1))
    print("Number of models in set 2: %d" % len(all_models2))
    print("Number of models overlapping: %d" % len(intersection))
    print("Overlapping models: %s" % intersection)
