import os
import shutil
import glob

root_dir = '/data3/honglinc/tdw_datasets/playroom_mv_v0/'
# datasets = ['archviz_house_split_0', 'archviz_house_split_1', 'archviz_house_split_2', 'archviz_house_split_3',
#             'mm_craftroom_1b_split_0', 'mm_craftroom_1b_split_1', 'mm_craftroom_1b_split_2', 'mm_craftroom_1b_split_3',
#             'tdw_room_split_0', 'tdw_room_split_1', 'tdw_room_split_2', 'tdw_room_split_3',
#             'box_split_0', 'box_split_1', 'box_split_2', 'box_split_3'
#             ]

datasets = ['archviz_house_split_val', 'mm_craftroom_1b_split_val', 'tdw_room_split_val', 'box_split_val']

datasets = [os.path.join(root_dir, d) for d in datasets]
save_dir = '/ccn2/u/honglinc/tdw_playroom_v0_val'
start_idx = 0
end_idx = 250

count = start_idx * len(datasets)

for i in range(start_idx, end_idx):
    for j, ds in enumerate(datasets):
        file_list = glob.glob(os.path.join(ds, 'sc{:04d}*'.format(i)))

        assert len(file_list) == 29, (ds, i)
        new_id = i * len(datasets) + j
        assert new_id == count
        count += 1
        src_files = sorted(file_list)
        dst_files = [x.replace('sc{:04d}'.format(i), 'sc{:06d}'.format(new_id)).replace(ds, save_dir) for x in src_files]
        for sc,ds in zip(src_files, dst_files):
            print(i, new_id, '{0: <100}'.format(sc), ds)
            shutil.copyfile(sc, ds)


print('Total files: ', count)