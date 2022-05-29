import os
import shutil
import glob
import random


base_dataset = 'tdw_30obj_manip_box'
bg_datasets = ['tdw_30obj_manip_tdw_room', 'tdw_30obj_manip_craft1b']
move_dataset = 'tdw_30obj_manip_box_move'
save_dir = 'tdw_30obj_manip'
start_idx = 0
end_idx = 10

count = start_idx
shutil.copyfile(os.path.join(base_dataset, 'metadata.json'), os.path.join(save_dir, 'metadata.json'))

for i in range(start_idx, end_idx):
    base_file_list = sorted(glob.glob(os.path.join(base_dataset, 'sc{:04d}*'.format(i))))

    select_bg_dataset = random.choice(bg_datasets)
    select_bg_idx = random.randint(start_idx, end_idx)
    print(select_bg_idx)

    for base_path in base_file_list:
        print(base_path, base_path.replace(base_dataset, save_dir))
        shutil.copyfile(base_path, base_path.replace(base_dataset, save_dir))
        if '.png' in base_path:
            # get changed background

            sc_changed_bg_path = base_path.replace(base_dataset, select_bg_dataset)
            dst_changed_bg_path = sc_changed_bg_path.replace('.png', '_changed.png').replace(select_bg_dataset, save_dir)
            print(sc_changed_bg_path, dst_changed_bg_path)
            shutil.copyfile(sc_changed_bg_path, dst_changed_bg_path)

            # get provide background
            sc_provided_bg_path = sc_changed_bg_path.replace('sc{:04d}'.format(i), 'sc{:04d}'.format(select_bg_idx))
            dst_provided_bg_path = sc_changed_bg_path.replace('.png', '_providing_bg.png').replace(select_bg_dataset, save_dir)
            print(sc_provided_bg_path, dst_provided_bg_path)
            shutil.copyfile(sc_provided_bg_path, dst_provided_bg_path)

            # get moved
            sc_mv_path = base_path.replace(base_dataset, move_dataset)
            dst_mv_path = sc_mv_path.replace('.png', '_moved.png').replace(move_dataset, save_dir)
            print(sc_mv_path, dst_mv_path)
            shutil.copyfile(sc_mv_path, dst_mv_path)

            print('-----')











print('Total files: ', count)