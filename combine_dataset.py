import os
import shutil
import glob


datasets = ['tdw_30obj_c1b', 'tdw_30obj_tdw_room', 'tdw_30obj_box']
save_dir = 'tdw_30obj_multibg'
start_idx = 1000
end_idx = 1200

count = start_idx * len(datasets)
for i in range(start_idx, end_idx):
    for j, ds in enumerate(datasets):
        file_list = glob.glob(os.path.join(ds, 'sc{:04d}*'.format(i)))
        assert len(file_list) == 29
        new_id = i * len(datasets) + j
        assert new_id == count
        count += 1
        src_files = sorted(file_list)
        dst_files = [x.replace('sc{:04d}'.format(i), 'sc{:04d}'.format(new_id)).replace(ds, save_dir) for x in src_files]
        for sc,ds in zip(src_files, dst_files):
            print(i, new_id, '{0: <50}'.format(sc), ds)
            shutil.copyfile(sc, ds)


print('Total files: ', count)