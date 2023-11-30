import os
import shutil
import glob

root_dir = ['tdw_room_more_bg', 'tdw_room_more_bg+1', 'tdw_room_more_bg_2', 'tdw_room_more_bg_3', 'tdw_room_more_bg_4']

datasets = []
for dir in root_dir:
    dataset = glob.glob(os.path.join(dir, '*'))
    for d in dataset:
        if 'generation.log' not in d:
            datasets.append(d)


save_dir = '/ccn2/u/honglinc/datasets/tdw_playroom_v3_more_bg'
start_idx = 0
end_idx = 100

count = 0

for i in range(start_idx, end_idx):
    for j, ds in enumerate(datasets):
        file_list = glob.glob(os.path.join(ds, 'sc{:04d}*'.format(i)))

        if len(file_list) != 29:
            continue
        assert len(file_list) == 29, (ds, i)

        new_id = count

        count += 1
        src_files = sorted(file_list)
        dst_files = [x.replace('sc{:04d}'.format(i), 'sc{:06d}'.format(new_id)).replace(ds, save_dir) for x in src_files]
        for sc,ds in zip(src_files, dst_files):
            print(i, new_id, '{0: <100}'.format(sc), ds)
            shutil.copyfile(sc, ds)


print('Total files: ', count)
